"""Project GPT-6 cost / latency of a protocol from a run directory.

Works on (a) a real OpenAI run (uses the logged usage and latencies) or (b) a dry run of another backend made
with --dry-run-payload (uses the logged request sizes of every call). For (b) all numbers are ESTIMATES:
  input tokens per call  = text_chars / 4 + n_images * 256   (512 px images; image accounting is an assumption)
  cached tokens per call = input tokens of the previous call in the same attempt (append-only prefix) --
                           the first call of an attempt is counted as uncached (conservative)
  output tokens per call = --out-tokens (reasoning + visible; unknown for GPT-6 Astra, default 1500)
  latency per call       = --latency (seconds; RoboICL measured ~30-50 s per GPT-6 Astra call at 50-90K input
                           tokens and high/xhigh effort -- treat as an upper-range planning number)
The number of model calls of a scripted run is a LOWER bound for an LLM agent (no perception / checking calls).

  python scripts/agent_cost.py runs/agent/l5_slide_to_target/<run> --out-tokens 1500 --latency 20 --calls-scale 2
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lffbench.agent.cost import PRICES, call_cost  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--out-tokens", type=float, default=1500)
    ap.add_argument("--latency", type=float, default=20.0)
    ap.add_argument("--calls-scale", type=float, default=1.0, help="multiply the call count (LLM vs scripted)")
    a = ap.parse_args()
    ev = [json.loads(l) for l in open(os.path.join(a.run_dir, "events.jsonl"))]
    calls = [e for e in ev if e["type"] == "llm_call"]
    runs = {e["seed"] for e in ev if e["type"] == "run_start"}
    real = [c for c in calls if c.get("usage")]
    if real and not any("dry_run_request" in c for c in calls):
        cost = sum(call_cost(c["usage"], a.model) or 0 for c in real)
        lat = [c["latency_s"] for c in real]
        u = defaultdict(int)
        for c in real:
            for kk, v in c["usage"].items():
                u[kk] += v
        out = dict(mode="measured", runs=len(runs), calls=len(real), calls_per_run=len(real) / max(len(runs), 1),
                   usage=dict(u), cost_usd=cost, cost_per_run=cost / max(len(runs), 1),
                   latency_mean_s=sum(lat) / len(lat), api_time_per_run_s=sum(lat) / max(len(runs), 1))
    else:
        prev = {}
        tot_in = tot_cached = 0
        cost1 = 0.0  # priced per call (the long-context surcharge applies per request)
        for c in calls:
            d = c.get("dry_run_request")
            if not d:
                continue
            key = (c["seed"], c["attempt"])
            tin = d["est_input_tokens"]
            cached = min(prev.get(key, 0), tin)
            prev[key] = tin
            tot_in += tin
            tot_cached += cached
            cost1 += call_cost(dict(input_tokens=tin, cached_tokens=cached, output_tokens=a.out_tokens), a.model)
        n = len(calls) * a.calls_scale
        scale = a.calls_scale
        usage = dict(input_tokens=tot_in * scale, cached_tokens=tot_cached * scale, output_tokens=n * a.out_tokens)
        cost = cost1 * scale
        out = dict(mode="estimate (dry run)", runs=len(runs), calls=n, calls_per_run=n / max(len(runs), 1),
                   input_tokens_per_call=tot_in / max(len(calls), 1), cached_fraction=tot_cached / max(tot_in, 1),
                   usage=usage, cost_usd=cost, cost_per_run=cost / max(len(runs), 1),
                   api_time_per_run_s=n * a.latency / max(len(runs), 1), prices=PRICES.get(a.model),
                   assumptions=dict(out_tokens_per_call=a.out_tokens, latency_s_per_call=a.latency,
                                    calls_scale=a.calls_scale, image_tokens_512px=256))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
