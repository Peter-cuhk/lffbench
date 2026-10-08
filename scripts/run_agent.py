"""Run an agent backend on an LFF-Bench task and log everything.

  source run_env.sh && export OMP_NUM_THREADS=1
  python scripts/run_agent.py --task l5_slide_to_target --backend scripted --memory full --feedback F2 --n 3 --k 5
  python scripts/run_agent.py --task l5_slide_to_target --backend openai --effort medium --memory full --n 20 --k 5

Writes runs/agent/<task>/<timestamp>_<backend>_<memory>_<feedback>/ with config.json, events.jsonl
(every LLM call with tokens + latency, every tool call with its report, every attempt's outcome and
feedback), images/ (every image the model saw, as the exact JPEG sent) and summary.json (succ@k per run,
first-success attempt, ΔICL fields, tokens / cost / latency).
"""
import argparse
import importlib
import json
import os
import pkgutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_tasks():
    import lffbench.tasks as task_pkg
    from lffbench.task_base import TASKS
    for m in pkgutil.iter_modules(task_pkg.__path__):
        try:
            importlib.import_module(f"lffbench.tasks.{m.name}")
        except Exception as e:  # other people's task files may be mid-edit; do not let them break this run
            print(f"[warn] could not import lffbench.tasks.{m.name}: {e!r}", file=sys.stderr)
    return TASKS


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--backend", choices=["scripted", "openai", "chat"], default="scripted")
    ap.add_argument("--memory", choices=["none", "full", "last", "mismatched"], default="full")
    ap.add_argument("--feedback", choices=["F0", "F1", "F2"], default="F2")
    ap.add_argument("--n", type=int, default=3, help="number of instances (runs)")
    ap.add_argument("--k", type=int, default=None, help="attempts per run (default: task.max_attempts)")
    ap.add_argument("--seed0", type=int, default=1000, help="first instance seed (dev seeds 1000+, use 5000+ for test)")
    ap.add_argument("--seeds", default=None, help="explicit comma-separated seeds (overrides --seed0/--n)")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "agent"))
    ap.add_argument("--tag", default="")
    # harness
    ap.add_argument("--instruction", choices=["direct", "indirect"], default="direct")
    ap.add_argument("--privileged-locate", action="store_true", help="ablation: add locate(object_name)")
    ap.add_argument("--image-res", type=int, default=512)
    ap.add_argument("--keyframe-res", type=int, default=None)
    ap.add_argument("--keyframes", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=30, help="tool calls per attempt")
    ap.add_argument("--no-auto-obs", action="store_true", help="send images only on get_observation")
    ap.add_argument("--grasp-report", choices=["bool", "name", "none"], default="bool")
    ap.add_argument("--early-stop", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--no-save-images", action="store_true")
    # mismatched memory
    ap.add_argument("--donor-run", default=None, help="run dir whose attempts serve as mismatched histories")
    ap.add_argument("--donor-offset", type=int, default=10000,
                    help="without --donor-run: donor instance seed = seed + offset, generated with memory=full")
    # scripted
    ap.add_argument("--scripted-policy", choices=["adaptive", "naive", "oracle", "blind"], default="adaptive")
    ap.add_argument("--dry-run-payload", action="store_true",
                    help="for non-openai backends: also build the GPT-6 request for every call and log its size")
    ap.add_argument("--dump-requests", type=int, default=0,
                    help="write the first N OpenAI requests (images elided) to requests/ for inspection")
    # openai
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--effort", default="medium", help="reasoning effort: low|medium|high|xhigh|max")
    ap.add_argument("--reasoning-summary", default=None, help="e.g. auto / concise / detailed (may need org verification)")
    ap.add_argument("--image-detail", default="auto", help="low|high|original|auto")
    ap.add_argument("--max-output-tokens", type=int, default=None)
    ap.add_argument("--service-tier", default=None)
    ap.add_argument("--prompt-cache-retention", default=None, help="in-memory | 24h")
    ap.add_argument("--max-live-images", type=int, default=16)
    ap.add_argument("--extra-body", default=None, help="JSON merged into the request body (new API fields)")
    ap.add_argument("--cache-breakpoints", choices=["none", "history", "history+last"], default="history+last",
                    help="explicit prompt_cache_breakpoint markers (set none if a gateway rejects the field)")
    ap.add_argument("--prompt-cache-options", default=None, help='JSON, e.g. {"ttl": "30m"}')
    # chat (OpenAI-compatible local server)
    ap.add_argument("--chat-base-url", default=None)
    ap.add_argument("--chat-model", default="qwen3-vl-8b")
    ap.add_argument("--chat-max-tokens", type=int, default=1024)
    return ap.parse_args(argv)


def redact_request(req):
    """Copy of a request with base64 images replaced by a short marker."""
    def red(o):
        if isinstance(o, dict):
            return {k: (f"<data url, {len(v)} chars>" if k in ("image_url", "url") and isinstance(v, str)
                        and v.startswith("data:") else red(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [red(v) for v in o]
        if isinstance(o, str) and len(o) > 2000 and o.startswith("gAAAA"):  # encrypted reasoning
            return f"<encrypted, {len(o)} chars>"
        return o
    return red(req)


def build_backend(args):
    from lffbench.agent.backends import make_backend
    if args.backend == "scripted":
        return make_backend("scripted", policy=args.scripted_policy)
    if args.backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY is not set (and OPENAI_BASE_URL if you use a proxy / gateway).")
        return make_backend("openai", model=args.model, reasoning_effort=args.effort,
                            reasoning_summary=args.reasoning_summary, image_detail=args.image_detail,
                            max_output_tokens=args.max_output_tokens, service_tier=args.service_tier,
                            prompt_cache_retention=args.prompt_cache_retention, max_live_images=args.max_live_images,
                            jpeg_quality=args.jpeg_quality, cache_breakpoints=args.cache_breakpoints,
                            prompt_cache_options=json.loads(args.prompt_cache_options) if args.prompt_cache_options
                            else None, extra_body=json.loads(args.extra_body) if args.extra_body else None)
    return make_backend("chat", model=args.chat_model, base_url=args.chat_base_url, max_tokens=args.chat_max_tokens,
                        max_live_images=min(args.max_live_images, 4), jpeg_quality=args.jpeg_quality)


def main(argv=None):
    args = parse(argv)
    from lffbench.agent.harness import AgentHarness, HarnessConfig, config_dict
    from lffbench.agent.metrics import aggregate
    from lffbench.agent.replay import derange, load_records
    from lffbench.agent.runlog import RunLogger

    tasks = load_tasks()
    if args.task not in tasks:
        sys.exit(f"unknown task {args.task}; known: {sorted(tasks)}")
    backend = build_backend(args)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(range(args.seed0, args.seed0 + args.n))
    cfg = HarnessConfig(memory=args.memory, feedback=args.feedback, k=args.k, image_res=args.image_res,
                        keyframe_res=args.keyframe_res, n_keyframes=args.keyframes, auto_obs=not args.no_auto_obs,
                        max_steps=args.max_steps, privileged_locate=args.privileged_locate,
                        grasp_report=args.grasp_report, instruction=args.instruction, early_stop=args.early_stop,
                        jpeg_quality=args.jpeg_quality, save_images=not args.no_save_images,
                        model_for_cost=args.model if args.backend == "openai" else "none")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{args.backend}{'-' + args.scripted_policy if args.backend == 'scripted' else ''}_{args.memory}_" \
           f"{args.feedback}" + (f"_{args.tag}" if args.tag else "")
    run_dir = os.path.join(args.out, args.task, name)
    os.makedirs(run_dir, exist_ok=True)
    log = RunLogger(run_dir, save_images=cfg.save_images, jpeg_quality=cfg.jpeg_quality)

    t0 = time.time()
    task = tasks[args.task]()
    print(f"[run_agent] env built in {time.time() - t0:.1f}s -> {run_dir}", flush=True)
    k = args.k or task.max_attempts

    probe = None
    if args.backend == "openai":
        probe = None
    elif args.dry_run_payload or args.dump_requests:
        from lffbench.agent.backends.openai_responses import OpenAIResponsesBackend
        probe = OpenAIResponsesBackend(model=args.model, reasoning_effort=args.effort, image_detail=args.image_detail,
                                       max_live_images=args.max_live_images, jpeg_quality=args.jpeg_quality,
                                       cache_breakpoints=args.cache_breakpoints, client=object())
    if args.dump_requests:
        os.makedirs(os.path.join(run_dir, "requests"), exist_ok=True)
        dumped = [0]
        target = backend if args.backend == "openai" else probe
        orig = target.build_request

        def build_and_dump(conv, tools, _orig=orig):
            req = _orig(conv, tools)
            if dumped[0] < args.dump_requests:
                dumped[0] += 1
                with open(os.path.join(run_dir, "requests", f"request_{dumped[0]:03d}.json"), "w") as f:
                    json.dump(redact_request(req), f, indent=1)
            return req
        target.build_request = build_and_dump

    log.write_json("config.json", dict(argv=sys.argv, args=vars(args), harness=config_dict(cfg),
                                       backend=backend.config(), task=dict(name=task.name, level=task.level,
                                                                           protocol=task.protocol, k=k,
                                                                           capabilities=list(task.capabilities))))
    harness = AgentHarness(task, backend, cfg, log, payload_probe=probe if args.backend != "openai" else None)

    donors = {}
    if args.memory == "mismatched":
        if args.donor_run:
            pool = load_records(args.donor_run, task=task, feedback=args.feedback)
            m = derange(seeds, pool)
            donors = {s: pool[m[s]] for s in seeds}
        else:
            dlog = RunLogger(os.path.join(run_dir, "donors"), save_images=True, jpeg_quality=cfg.jpeg_quality)
            dcfg = HarnessConfig(**{**config_dict(cfg), "memory": "full", "early_stop": "auto"})
            dh = AgentHarness(task, backend, dcfg, dlog)
            for s in seeds:
                _, recs = dh.run_instance(s + args.donor_offset)
                donors[s] = recs
                print(f"[donor] seed={s + args.donor_offset} first_success="
                      f"{next((r.attempt for r in recs if r.success), None)}", flush=True)
            dlog.close()

    runs = []
    for i, s in enumerate(seeds):
        summ, recs = harness.run_instance(s, donor=donors.get(s))
        runs.append(summ)
        print(f"[{i + 1}/{len(seeds)}] seed={s} first_success={summ['first_success']} attempts={summ['n_attempts']} "
              f"status={summ['status']} llm_calls={summ['n_llm_calls']} tool_calls={summ['n_tool_calls']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    agg = aggregate(runs, k)
    log.write_json("summary.json", dict(task=args.task, backend=backend.config(), memory=args.memory,
                                        feedback=args.feedback, k=k, seeds=seeds, aggregate=agg,
                                        runs=[{kk: v for kk, v in r.items() if not kk.startswith("_")} for r in runs]))
    log.close()
    print(json.dumps({kk: agg[kk] for kk in ("n_runs", "n_error_runs", "succ_at", "attempt1_success",
                                             "mean_first_success", "pass_at_k_iid_analytic", "repeated_action_rate",
                                             "llm_calls_per_run", "tool_calls_per_run", "sim_wall_per_run_s")},
                     default=str))
    print(f"[run_agent] done -> {run_dir}")
    return run_dir, agg, runs


if __name__ == "__main__":
    main()
