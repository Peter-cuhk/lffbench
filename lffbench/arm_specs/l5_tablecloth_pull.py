"""Controlled arms for l5_tablecloth_pull (docs/14 §6).

Hidden variable: mu, the cube's friction (cube-board and cube-table contacts; apply_instance() sets it on every reset,
oracle_params() bisects the pull speed with it). The layout keys (board_xy, band_d) are left alone: band_d was sampled
from the original mu, but it is the visible band position, so H0 and the twin keep it and move only mu.

Physics (measured, 2026-10-12): the cube's displacement toward the robot grows with mu and shrinks with the pull speed
up to ~0.8 m/s (0.8 and 1.0 m/s give identical results: the controller saturates). The naive policy yanks at 0.6 m/s.
Every original instance lands SHORT of the band (cube forward of it, err -1.6 .. -3.5 cm): the lesson is "pull
slower".

  h0     mu at which the 0.6 m/s yank lands the cube on the band's centre line (calibrated per full5 seed by a
         simulated sweep; a linear fit in band_d for other seeds).
  twin   the opposite trap: mu high enough that the 0.6 m/s yank drags the cube PAST the band toward the robot
         (err +1.5 .. +2.7 cm), so the lesson is "pull faster" (the twin's oracle pulls at 0.64-0.79 m/s; agents can
         command up to 1.0 m/s). Applying that lesson to the original makes its cube land even further short.
         The window is narrow: above it the cube cannot be stopped in the band even at the saturated top speed, so the
         twin's naive error is only ~1.5-2.7 cm (similar size to the original's short-fall) and the twin's oracle is
         within 0.2-0.5 cm of the band tolerance on some seeds. Calibrated per seed; for other seeds 1.2 x the H0 mu.
  reveal states the cube's friction coefficient (the only fact that differs between instances).
"""

HIDDEN_KEYS = ("mu",)

# Calibrated 2026-10-12 with the scripted policies (scratchpad arms/fine.py; deterministic sim).
#   seed: (h0 mu, naive err at h0 [cm]), (twin mu, naive err [cm], twin oracle speed [m/s], twin oracle err [cm])
_CAL = {
    3000: dict(h0=0.1543, twin=0.1876),  # h0 +0.08; twin naive +1.65, oracle 0.719 m/s +0.60
    3001: dict(h0=0.1744, twin=0.1995),  # h0 +0.13; twin naive +2.17, oracle 0.638 m/s +0.47
    3002: dict(h0=0.1393, twin=0.1716),  # h0 -0.11; twin naive +1.48, oracle 0.759 m/s +0.78
    3003: dict(h0=0.1442, twin=0.1775),  # h0 +0.08; twin naive +2.18, oracle 0.759 m/s +0.44
    3004: dict(h0=0.1944, twin=0.2250),  # h0 +0.06; twin naive +2.74, oracle 0.790 m/s +0.51
}


def _h0_mu(inst):
    c = _CAL.get(int(inst.get("seed", -1)))
    return c["h0"] if c else 0.0835 + 1.485 * float(inst["band_d"])


def h0(inst):
    return {"mu": float(_h0_mu(inst))}


def twin(inst):
    c = _CAL.get(int(inst.get("seed", -1)))
    return {"mu": float(c["twin"] if c else 1.2 * _h0_mu(inst))}


def reveal(inst):
    return (f"The red cube's friction coefficient (against the board and against the table) is about "
            f"{float(inst['mu']):.3f}.")
