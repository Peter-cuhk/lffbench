"""Run one robot_server session (style la) with one agent harness through its adapter (lffbench/harnesses, docs/15).

  python scripts/harness_run.py --harness claude_code --session runs/<...>/session.json --model claude-sonnet-5-5 --effort high
  python scripts/harness_run.py --harness codex       --session ... --model gpt-6-astra --effort high
  python scripts/harness_run.py --harness ours        --session ... --provider anthropic --model ... --effort high

Harnesses: ours (scripts/api_player.py, unchanged), claude_code (arm B, --bare), claude_code_full (arm B', no --bare),
codex (arm F). Native harnesses get the reference driver's instructions + first status report as their first user
message (byte-identical README), only our MCP server (scripts/mcp_robot_server.py) as tools, a one-time HOME / config
dir, an opaque work dir /tmp/lffrun/<12 hex> and a process environment that is exactly a whitelist.

API variables (never inherited from the calling Claude Code / Codex session):
  claude_code*: key LFF_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY; endpoint --api-base-url or LFF_ANTHROPIC_BASE_URL
                (default: the CLI's own default, the official API)
  codex:        key LFF_OPENAI_API_KEY or OPENAI_API_KEY; endpoint --api-base-url, LFF_OPENAI_BASE_URL or
                OPENAI_BASE_URL (as api_player), else https://api.openai.com/v1
  ours:         as api_player.py (its own environment)

Writes into the run's log dir: contract.json, first_message.txt, harness.json (harness name, version, argv with the
first message replaced by its sha256, env key names, contract_sha256, first_message_sha256, config_sha256, preflight
result), harness_preflight.json (full normalised fingerprint), the native event stream, trajectory.jsonl,
harness_audit.json. Exit codes: 0 done, 2 refused at launch (fail-closed), 3 harness error, 143 stopped.
"""
import argparse
import json
import os
import signal
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lffbench.harnesses import PROVIDER, EpisodeSpec, HarnessRefused, get_adapter, render_first_message  # noqa: E402
from lffbench.harnesses import contract as C  # noqa: E402


def api_env_for(harness, base_url):
    """-> (variables passed to the harness, endpoint or None, endpoint label for harness.json). Endpoints are never
    written to the logs, only where they came from."""
    if harness.startswith("claude_code"):
        e = {}
        key = os.environ.get("LFF_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if key:
            e["ANTHROPIC_API_KEY"] = key
        url = base_url or os.environ.get("LFF_ANTHROPIC_BASE_URL")
        if url:
            e["ANTHROPIC_BASE_URL"] = url
        return e, url, ("--api-base-url" if base_url else "LFF_ANTHROPIC_BASE_URL") if url else "CLI default (official)"
    if harness == "codex":
        key = os.environ.get("LFF_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        src = ("--api-base-url" if base_url else "LFF_OPENAI_BASE_URL" if os.environ.get("LFF_OPENAI_BASE_URL")
               else "OPENAI_BASE_URL (same as api_player)" if os.environ.get("OPENAI_BASE_URL") else "official default")
        url = base_url or os.environ.get("LFF_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        return ({"OPENAI_API_KEY": key} if key else {}), url, src
    return {}, None, "api_player's own environment"


def write_json(path, d):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def run(a, adapter_kw=None):
    """Returns (exit code, harness.json dict). adapter_kw overrides adapter construction (self-check)."""
    provider = a.provider or PROVIDER.get(a.harness, "")
    ep = EpisodeSpec.from_session(a.session, a.model, a.effort, provider)
    contract, csha = C.write_contract(ep.log_dir)
    if adapter_kw is None:
        api_env, url, endpoint = api_env_for(a.harness, a.api_base_url)
        adapter_kw = dict(timeout_s=a.timeout)
        if a.harness.startswith("claude_code"):
            adapter_kw.update(api_env=api_env, max_budget_usd=a.max_budget_usd, allow_unpinned=a.allow_unpinned)
        elif a.harness == "codex":
            adapter_kw.update(api_env=api_env, base_url=url, allow_unpinned=a.allow_unpinned)
        else:
            adapter_kw.update(provider=provider)
    else:
        endpoint = "set by the caller (self-check)"
    ad = get_adapter(a.harness, **adapter_kw)
    ver = ad.version()
    hj_path = os.path.join(ep.log_dir, "harness.json")
    hj = dict(harness=ad.name, version=ver.get("version"), binary=ver.get("binary"), model=a.model, effort=a.effort,
              provider=provider, capabilities={k: vars(c) for k, c in ad.capabilities.items()},
              contract_sha256=csha, task=ep.task, seed=ep.seed, api_endpoint_from=endpoint, status="preparing",
              t_prepare=time.time())
    if ad.name == "ours":
        first = None
        spec = ad.prepare(ep)
    else:
        first, _ = render_first_message(ep)
        with open(os.path.join(ep.log_dir, "first_message.txt"), "w") as f:
            f.write(first)
        spec = ad.prepare(ep, first, **(dict(reuse_state=a.reuse_state) if getattr(a, "reuse_state", None) else {}))
        hj["first_message_sha256"] = C.sha256_text(first)
    hj.update(argv=ad.logged_argv(spec), cwd=spec.cwd, env_keys=sorted(spec.env),
              env_mode="exact whitelist (Popen env=, same as env -i)" if ad.name != "ours" else "inherited (own code)",
              state_dirs=spec.state_dirs, config_sha256=ad.config_sha256(spec, ep), status="launching")
    write_json(hj_path, hj)
    h, code = None, 0
    stop = dict(flag=False)

    def on_term(*_):
        stop["flag"] = True
        if h is not None and h.proc is not None:
            ad.kill(h)
    signal.signal(signal.SIGTERM, on_term)
    try:
        try:
            h = ad.launch(spec, ep, skip_preflight=getattr(a, "skip_preflight", False))
        except HarnessRefused as e:
            pre = dict(spec.preflight or {})
            hj.update(status="refused", refused=str(e), preflight={k: v for k, v in pre.items() if k != "fingerprint"})
            write_json(hj_path, hj)
            print("REFUSED:", e, flush=True)
            return 2, hj
        pre = dict(spec.preflight or {})
        if "fingerprint" in pre:
            write_json(os.path.join(ep.log_dir, "harness_preflight.json"), pre)
        hj.update(status="running", pid=h.pid, env_observed_keys=h.env_observed_keys, t_start=h.t_start,
                  preflight={k: v for k, v in pre.items() if k != "fingerprint"}, init=h.info.get("init"))
        write_json(hj_path, hj)
        ad.wait(h)
        if stop["flag"]:
            h.exit_reason, code = "stopped", 143
        tw = ad.collect(h)
        au = ad.audit(h)
        if ad.name == "ours" and spec.first_message:
            hj["first_message_sha256"] = C.sha256_text(spec.first_message)
        hj.update(status="done" if code == 0 else "stopped", exit_code=h.exit_code, exit_reason=h.exit_reason,
                  t_end=h.t_end, wall_s=round(h.t_end - h.t_start, 1), audit_ok=au.get("ok"),
                  violations=au.get("violations"), final_metrics=(tw.final or {}).get("final_metrics"))
        if code == 0 and h.exit_code not in (0, None):
            code = 3
        write_json(hj_path, hj)
        return code, hj
    except Exception as e:
        hj.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
        write_json(hj_path, hj)
        if h is not None and h.proc is not None and h.proc.poll() is None:
            ad.kill(h)
        return 3, hj
    finally:
        if not getattr(a, "keep_run_dir", False):
            ad.cleanup(spec)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True, choices=["ours", "claude_code", "claude_code_full", "codex"])
    ap.add_argument("--session", required=True)
    ap.add_argument("--provider", default=None, help="ours: openai / anthropic / gemini / chat")
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--api-base-url", default=None)
    ap.add_argument("--max-budget-usd", type=float, default=None, help="claude_code*: --max-budget-usd")
    ap.add_argument("--timeout", type=float, default=10800)
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="launch even if no pin exists for this version / model / effort (the preflight still runs)")
    ap.add_argument("--keep-run-dir", action="store_true", help="do not delete the one-time dirs (debugging)")
    return ap


def main():
    code, hj = run(parser().parse_args())
    print(json.dumps({k: hj.get(k) for k in ("harness", "status", "exit_reason", "audit_ok", "violations",
                                             "contract_sha256", "final_metrics")}, default=str), flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
