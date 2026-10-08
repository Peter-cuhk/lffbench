"""Run-level metrics (design doc 02 §5): succ@k per run, attempt-1 success, first-success attempt,
fields for ΔICL = succ@k(with history) - pass@k_iid(independent retries), same-mistake (repeated
action) rate, residuals, tokens / cost / latency."""
import numpy as np

from .context import wilson


def run_summary(seed, records, k, status="ok", llm_calls=(), extra=None):
    """records: list[AttemptRecord] of one run (one instance)."""
    succ = [bool(r.success) for r in records]
    first = next((r.attempt for r in records if r.success), None)
    fails = [r for r in records if not r.success]
    pairs = [(a, b) for a, b in zip(records, records[1:]) if not a.success]
    repeats = [a.action_signature() == b.action_signature() for a, b in pairs]
    residuals = [{kk: v for kk, v in r.outcome.items() if kk.endswith("_err") and isinstance(v, (int, float))}
                 for r in records]
    lat = [c.get("latency_s", 0.0) for c in llm_calls]
    usage = {key: int(sum((c.get("usage") or {}).get(key, 0) or 0 for c in llm_calls))
             for key in ("input_tokens", "cached_tokens", "cache_write_tokens", "output_tokens",
                         "reasoning_tokens")}
    cost = [c.get("cost_usd") for c in llm_calls if c.get("cost_usd") is not None]
    out = dict(seed=seed, status=status, k=k, n_attempts=len(records), success_by_attempt=succ,
               first_success=first, success=first is not None, attempt1_success=bool(succ[0]) if succ else False,
               ended_by=[r.ended_by for r in records], n_failed_attempts=len(fails),
               repeated_action_pairs=int(sum(repeats)), failed_pairs=len(pairs), residuals=residuals,
               n_llm_calls=len(llm_calls), llm_latency_s=float(sum(lat)), usage=usage,
               cost_usd=float(sum(cost)) if cost else None,
               n_tool_calls=int(sum(len(r.steps) for r in records)))
    if extra:
        out.update(extra)
    return out


def aggregate(runs, k):
    ok = [r for r in runs if r["status"] == "ok"]
    n = len(ok)
    first = [r["first_success"] for r in ok]
    curve, ci = [], []
    for j in range(1, k + 1):
        s = sum(1 for f in first if f is not None and f <= j)
        curve.append(s / n if n else float("nan"))
        ci.append(wilson(s, n))
    p1 = curve[0] if curve else float("nan")
    succ_first = [f for f in first if f is not None]
    calls = [c for r in ok for c in r.get("_llm_calls", [])]
    lat = np.array([c.get("latency_s", 0.0) for c in calls]) if calls else np.zeros(0)
    pairs = sum(r["failed_pairs"] for r in ok)
    tot = {key: int(sum(r["usage"].get(key, 0) for r in ok))
           for key in ("input_tokens", "cached_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens")}
    costs = [r["cost_usd"] for r in ok if r.get("cost_usd") is not None]
    return dict(
        n_runs=n, n_error_runs=len(runs) - n, k=k,
        succ_at=curve, succ_at_ci95=ci, succ_at_k=curve[-1] if curve else float("nan"),
        attempt1_success=p1,
        # ΔICL needs pass@k of independent retries: measure it with --memory none on the same seeds and
        # compare with scripts/agent_compare.py. The analytic value assumes i.i.d. attempts with the
        # attempt-1 success rate of THIS run set (a cheaper but weaker reference).
        pass_at_k_iid_analytic=1 - (1 - p1) ** k if n else float("nan"),
        delta_icl_vs_analytic_iid=(curve[-1] - (1 - (1 - p1) ** k)) if n else float("nan"),
        mean_first_success=float(np.mean(succ_first)) if succ_first else None,
        median_first_success=float(np.median(succ_first)) if succ_first else None,
        repeated_action_rate=(sum(r["repeated_action_pairs"] for r in ok) / pairs) if pairs else None,
        llm_calls_per_run=float(np.mean([r["n_llm_calls"] for r in ok])) if ok else None,
        llm_latency_mean_s=float(lat.mean()) if lat.size else None,
        llm_latency_p50_s=float(np.percentile(lat, 50)) if lat.size else None,
        llm_latency_p95_s=float(np.percentile(lat, 95)) if lat.size else None,
        llm_latency_per_run_s=float(np.mean([r["llm_latency_s"] for r in ok])) if ok else None,
        sim_wall_per_run_s=float(np.mean([r.get("sim_wall_s", 0.0) for r in ok])) if ok else None,
        tool_calls_per_run=float(np.mean([r["n_tool_calls"] for r in ok])) if ok else None,
        usage_total=tot,
        cache_hit_rate=(tot["cached_tokens"] / tot["input_tokens"]) if tot["input_tokens"] else None,
        cost_usd_total=float(sum(costs)) if costs else None,
        cost_usd_per_run=float(np.mean(costs)) if costs else None,
    )


def paired_delta(with_runs, iid_runs, k, n_boot=10000, seed=0):
    """ΔICL = succ@k(with) - succ@k(iid) on the seeds both run sets share, with a paired bootstrap CI."""
    a = {r["seed"]: r for r in with_runs if r["status"] == "ok"}
    b = {r["seed"]: r for r in iid_runs if r["status"] == "ok"}
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return dict(n=0)

    def ok(r):
        return r["first_success"] is not None and r["first_success"] <= k

    x = np.array([ok(a[s]) for s in seeds], float)
    y = np.array([ok(b[s]) for s in seeds], float)
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(1)
    return dict(n=len(seeds), succ_at_k_with=float(x.mean()), succ_at_k_iid=float(y.mean()), delta_icl=float(d.mean()),
                ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                discordant=dict(with_only=int(((x == 1) & (y == 0)).sum()), iid_only=int(((x == 0) & (y == 1)).sum())))
