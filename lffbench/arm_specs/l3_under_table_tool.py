"""Controlled arms for l3_under_table_tool (2026-10-12).

Hidden variable: inst["pane_side"] -- the y side of the low glass table that is closed by an invisible clear acrylic
panel (+1 = robot's left, -1 = right; sampled as -side, i.e. the side away from the stick, where the bin stands).
apply_instance turns on the collision of the panel whose side equals pane_side (inst.get("pane_side", -side)); a
value matching neither side (0) leaves both panels off.

h0    pane_side = 0: no panel. The prior (slide the stick in and push the cube straight across, out towards the bin)
      gets the cube out on the bin's side, then grasps it and drops it in the bin.
twin  pane_side = side: the panel closes the OTHER y side -- the side the stick lies on -- instead of the bin's side.
      The prior still fails visibly, but differently: the stick's head runs into the panel at the table edge and
      the cube is not touched / stays under the glass (original: the cube is pushed across and stops against the
      panel at the bin's side). The blocked side is the opposite one. Measured (seeds 3000-3004): twin naive fails
      5/5 with F2 mode "blocked" (stick stopped at the panel, the hand ran into the glass edge, cube did not move).
      Limitation: with a single panel on a y side, the reference correction is the same in both (push the cube out
      of the open FRONT side from behind the table; oracle_params ignores the instance), so the cross check
      "twin oracle on the original" succeeds (measured 5/5). A twin whose correction fails on the original would need a second
      panel / panels on the front or back side plus a new scripted push (task-file change); not done.
reveal  Which side is closed, in the words of the task's F2 feedback.
"""
HIDDEN_KEYS = ("pane_side",)


def _pane_side(inst):
    return int(inst.get("pane_side", -inst["side"]))


def h0(inst):
    return {"pane_side": 0}


def twin(inst):
    return {"pane_side": -_pane_side(inst)}


def reveal(inst):
    ps = _pane_side(inst)
    lr = "the robot's left" if ps > 0 else "the robot's right"
    return (f"A clear acrylic panel, invisible in the camera images, closes the side of the low glass table on {lr} "
            f"(between its two legs, from the tabletop up to the glass); nothing can pass under the table edge on that "
            f"side.")
