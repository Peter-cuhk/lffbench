"""Controlled arms for l3_wall_flush (2026-10-12).

Hidden variable: inst["gap"] -- the box-wall gap, 0-3 mm (below camera resolution), far less than the ~1.2 cm a
finger needs, so the wall-side finger lands on the wall top. sample_instance caches the wall position derived from it
(wall_x = box_x + side * (wall half thickness + box half width + gap)) and apply_instance reads only inst["wall_xy"],
so every override below sets wall_xy as well.

h0    gap = 5 cm: box, plate, wall length and wall y unchanged, the wall moved 4.7-5 cm away from the box (on the
      same side). There is room for the whole finger and the prior (grasp across the width where the box lies) works.
      (2.5 cm was not enough: the open fingers reach 6.6 cm from the grasp centre, so with the wall on the far side
      the outer edge of the finger still caught the wall top and the box slipped out on the way, seeds 3000 / 3003.)
twin  Gap unchanged (box still flush, trap active), box / plate / wall length / wall x unchanged, but the wall slid
      along y so that the box sits near the OTHER end of the wall, 2 cm of wall beyond the box's end (the sampler's
      extreme). The original's nearer wall end is +y (or -y); the twin's is -y (or +y): the push that clears the
      wall goes the opposite way, and is the shortest possible push (about 10 + 4.5 cm).
      Note: pushing the original's box towards its FAR end also clears the wall (just a longer push), so the cross
      check "twin oracle on the original" succeeds (measured 5/5, seeds 3000-3004; results/l3_wall_flush/arms/
      validate.json). The twin changes which way to push, not whether pushing works.
reveal  The gap, and on which side of the box the wall is, in the task's F2 frame words.
"""
BOX_HALF_X, BOX_HALF_Y, WALL_HALF_T = 0.03, 0.08, 0.015
H0_GAP = 0.05

HIDDEN_KEYS = ("gap",)


def _wall_x(inst, gap):
    return inst["box_xy"][0] + inst["side"] * (WALL_HALF_T + BOX_HALF_X + gap)


def h0(inst):
    return {"gap": H0_GAP, "wall_xy": [_wall_x(inst, H0_GAP), inst["wall_xy"][1]]}


def twin(inst):
    wy, by = inst["wall_xy"][1], inst["box_xy"][1]
    u_max = inst["wall_hl"] - BOX_HALF_Y - 0.02
    nearer = 1 if by - wy > 0 else -1  # same rule as WallFlush._nearer_end
    u_twin = -nearer * u_max  # box offset from the wall centre in the twin: near the other end
    return {"wall_xy": [inst["wall_xy"][0], by - u_twin]}


def reveal(inst):
    side = "far side (away from the robot)" if inst["side"] > 0 else "near side (toward the robot)"
    return (f"The long box is pressed against the wall on its {side} along its whole length: the gap between box "
            f"and wall is only about {max(100 * inst['gap'], 0.1):.1f} cm.")
