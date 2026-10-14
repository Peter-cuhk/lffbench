"""Controlled arms for l4_fit_container (2026-10-12).

Hidden variable: inst["cap"] -- clear (unrendered) bumper caps on both ends of the phone, 0.8-0.9 cm each, so the
phone is 1.6-1.8 cm longer than it looks. The prior picks the box with the most visible room to spare ("snug"),
which is too short for the capped phone; only the "fit" box is long enough.

h0    cap = 0: no caps. The phone is exactly as big as it looks, the "snug" box the prior picks fits it (as do "fit"
      and "short_wide"). inst["fit"] is left as is (that box still fits).
twin  Same box positions / yaws, same phone, same caps (trap still active), but the inner sizes (and kinds) of the
      "fit" and "snug" boxes are swapped. In the twin the prior goes to the place where the ORIGINAL's fitting box
      stands (now with snug sizes) and fails on the rim; the twin's fitting box stands where the original's snug box
      stands. So the twin's lesson at the action level ("not that box, use that other box") is exactly the reverse
      of the original's: following it on the original means choosing the original's failing (snug) box.
      Caveat: the abstract lesson ("the phone is longer than it looks, take the box with the most length to spare")
      is the same in both -- a negative cap (phone shorter than it looks) is not physically possible here. The box
      sizes in the twin images differ from the original's at those two positions (they are visible).
reveal  The cap length, in cm, per end and in total.
"""
HIDDEN_KEYS = ("cap",)


def h0(inst):
    return {"cap": 0.0}


def _idx(inst, kind):
    return next(j for j, b in enumerate(inst["boxes"]) if b["kind"] == kind)


def twin(inst):
    f, s = _idx(inst, "fit"), _idx(inst, "snug")
    boxes = [dict(b) for b in inst["boxes"]]
    for k in ("kind", "inner"):
        boxes[f][k], boxes[s][k] = inst["boxes"][s][k], inst["boxes"][f][k]
    perc = [list(p) for p in inst["_perc"]]
    perc[f], perc[s] = list(inst["_perc"][s]), list(inst["_perc"][f])
    return {"boxes": boxes, "fit": s, "fit_crossed": boxes[s]["crossed"], "_perc": perc}


def reveal(inst):
    c = 100 * inst["cap"]
    return (f"The phone has clear bumper caps on both of its short ends that do not show in the camera images; "
            f"each cap adds about {c:.1f} cm, so the phone is about {2 * c:.1f} cm longer than it looks.")
