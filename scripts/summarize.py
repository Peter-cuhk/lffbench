"""Aggregate results/<task>/validate.json into results/SUMMARY.md (headroom table)."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*", "validate.json"))):
        r = json.load(open(p))
        s = r["succ_at_k"]

        def g(kind, j):
            v = s.get(kind)
            return "—" if v is None else f"{100 * v[j - 1]:.0f}"

        k = r["k"]
        rows.append(f"| {r['task']} | {r['level']} | {', '.join(r['capabilities'])} | {r['n']} | "
                    f"{g('oracle', 1)} | {g('naive', 1)} | {g('adaptive', 1)} / {g('adaptive', 2)} / {g('adaptive', k)} | "
                    f"{g('blind', k)} | {r.get('seconds', 0) / 60:.0f} min |")
    head = ("| task | level | capabilities | n | oracle@1 | naive@1 | adaptive @1/@2/@k | blind@k | wall |\n"
            "|---|---|---|---|---|---|---|---|---|")
    text = "# LFF-Bench headroom validation (scripted reference policies)\n\n" + head + "\n" + "\n".join(rows) + "\n"
    out = os.path.join(ROOT, "results", "SUMMARY.md")
    with open(out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
