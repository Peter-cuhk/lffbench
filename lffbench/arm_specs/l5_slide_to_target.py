"""Controlled arms for l5_slide_to_target (docs/14 §6).

Hidden variable: mu, the cube-table sliding friction (apply_instance() sets it on every reset; oracle_params() bisects
the strike speed with it). The layout (cube_xy, target_dist) is kept.

The naive strike is 0.30 m/s. Where the cube stops depends on mu and, by up to ~1.5 cm, on the cube's position on the
table, and the target tolerance is only +-1 cm, so H0 and the twin are calibrated per full5 seed with a simulated sweep
(deterministic sim, scratchpad arms/fine.py, 2026-10-12); other seeds fall back to a power-law fit of the stopping
distance at 0.30 m/s (dist ~ 0.00453 / mu^1.04 m).

  h0     mu at which the 0.30 m/s strike stops the cube on the square's centre (|err| <= 0.3 cm).
  twin   the opposite trap: if the original cube overshoots (slippery) the twin is grippy and stops short, and vice
         versa, so the correction is the other way (slow down <-> speed up). The twin's naive error is at least 2 cm
         and, where the twin's oracle still succeeds, about as large as the original's (mirrored). Seeds 3000 and 3007
         overshoot by 6.9 / 4.1 cm, but a grippy cube cannot be made to stop that far short and still be solvable,
         so their twins stop 2.1 / 3.4 cm short.
  reveal states the friction coefficient only.
"""

HIDDEN_KEYS = ("mu",)

# seed: h0 mu (naive err there), twin mu (twin naive err, twin oracle speed m/s -> err). Original naive err in brackets.
_CAL = {
    3000: dict(h0=0.0749, twin=0.1060),  # [orig +6.94 cm] h0 +0.26; twin -2.13, oracle 0.551 -> +0.01
    3001: dict(h0=0.0602, twin=0.0783),  # [orig +2.95 cm] h0 +0.14; twin -2.73, oracle 0.450 -> 0.00
    3003: dict(h0=0.0583, twin=0.0463),  # [orig -1.73 cm] h0 +0.03; twin +2.75, oracle 0.242 -> -0.13
    3004: dict(h0=0.0537, twin=0.0444),  # [orig -1.66 cm] h0 -0.31; twin +2.14, oracle 0.240 -> -0.02
    3007: dict(h0=0.0532, twin=0.0984),  # [orig +4.05 cm] h0 +0.15; twin -3.39, oracle 0.570 -> -0.18
}


def _h0_mu(inst):
    c = _CAL.get(int(inst.get("seed", -1)))
    return c["h0"] if c else (0.00453 / float(inst["target_dist"])) ** (1 / 1.04)


def h0(inst):
    return {"mu": float(_h0_mu(inst))}


def twin(inst):
    c = _CAL.get(int(inst.get("seed", -1)))
    if c:
        return {"mu": float(c["twin"])}
    m0 = _h0_mu(inst)  # fallback: mirror mu about the H0 value in log space, kept in a solvable range
    return {"mu": float(min(max(m0 * m0 / float(inst["mu"]), 0.03), 0.10))}


def reveal(inst):
    return f"The red cube's sliding friction coefficient on the table is about {float(inst['mu']):.3f}."
