"""ΔICL between two agent run directories on the same seeds.

  python scripts/agent_compare.py --with runs/agent/<task>/<full-memory run> --iid runs/agent/<task>/<memory-none run>

ΔICL = succ@k(with history) - pass@k_iid(independent retries), paired by instance seed, with a paired
bootstrap 95% interval. Also prints the analytic i.i.d. reference 1-(1-p1)^k from the 'with' run.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lffbench.agent.metrics import paired_delta  # noqa: E402


def load(d):
    with open(os.path.join(d, "summary.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with", dest="with_", required=True)
    ap.add_argument("--iid", required=True)
    ap.add_argument("--k", type=int, default=None)
    a = ap.parse_args()
    w, i = load(a.with_), load(a.iid)
    k = a.k or min(w["k"], i["k"])
    res = paired_delta(w["runs"], i["runs"], k)
    res["k"] = k
    res["with"] = dict(memory=w["memory"], feedback=w["feedback"], attempt1=w["aggregate"]["attempt1_success"],
                       pass_at_k_iid_analytic=w["aggregate"]["pass_at_k_iid_analytic"])
    res["iid"] = dict(memory=i["memory"], feedback=i["feedback"])
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
