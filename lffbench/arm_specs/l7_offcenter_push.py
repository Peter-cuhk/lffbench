"""Controlled arms for l7_offcenter_push (docs/14 §6).

Hidden variable: com_offset = e, the shift of the box's centre of mass along its long axis (world +y = robot's left;
|e| in [4.5, 7.5] cm). apply_instance() writes it into body_ipos on every reset; oracle_params() pushes at c = e.

  h0     e = 0: centre of mass in the middle, so the natural push at the middle of the back face goes straight.
  twin   e -> -e: same distance from the middle, other end heavy. The naive push turns the box just as much but the
         other way (clockwise <-> counterclockwise, the other end lags), and the right contact point is on the other
         side of the middle, so the original's lesson ("move the contact toward the left/right end") is reversed.
  reveal states |e| and its side in the robot-frame words of the F2 feedback (robot's left / right).
"""

HIDDEN_KEYS = ("com_offset",)


def h0(inst):
    return {"com_offset": 0.0}


def twin(inst):
    return {"com_offset": -float(inst["com_offset"])}


def reveal(inst):
    e = float(inst["com_offset"])
    side = "left" if e > 0 else "right"
    return (f"The box's centre of mass is not at its middle: it is about {100 * abs(e):.1f} cm from the middle along "
            f"the box's long side, toward the robot's {side}.")
