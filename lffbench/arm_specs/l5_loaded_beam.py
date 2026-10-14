"""Controlled arms for l5_loaded_beam (2026-10-12, docs/14 sec. 6).

Hidden variable: the bar's centre of mass `com_offset` (m along the bar from its middle, + = the robot's left end),
set at reset by l5_slippery_lift.set_hidden_weight (body_ipos + inertia + envs.refit_bounds, so a changed value is
honoured). `com_total` (bar + both blocks) is derived from it in sample_instance and read by oracle_params, so it is
overridden together. Mass, friction and layout stay as sampled.

h0   : centre of mass at the middle (0): the common-sense middle grasp lifts the bar level.
twin : the same offset on the other side (-com_offset): the middle grasp still tilts the bar and spills a block, but
       now towards the other end, so the right grasp point is mirrored.
reveal: the hidden weight's side and the bar's centre of mass in cm (plus the bar's mass in g), in the F2 words.
"""
HIDDEN_KEYS = ("com_offset", "com_total")


def h0(inst):
    return dict(com_offset=0.0, com_total=0.0)


def twin(inst):
    return dict(com_offset=-float(inst["com_offset"]), com_total=-float(inst["com_total"]))


def reveal(inst):
    e = float(inst["com_offset"])
    side = "left" if e > 0 else "right"
    return (f"The bar is not uniform: a heavy weight is hidden inside it towards its {side} end (the robot's {side}). "
            f"The bar weighs {1000 * inst['mass']:.0f} g and its centre of mass is {100 * abs(e):.1f} cm to the "
            f"{side} of its middle.")
