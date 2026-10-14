"""Controlled arms for l3_card_shelf (2026-10-12, docs/14 sec. 6).

Hidden variable: which end of the (symmetric-looking) card holds the 8 g weight. The centre of mass is compiled into
the card's MJCF along its local +x, so the heavy end is chosen purely by `card_yaw` (heavy end direction = yaw);
`heavy_near` / `heavy_edge` are bookkeeping only (nothing in reset / outcome / policies reads them), kept consistent.
No override changes geometry, mass or inertia, so no bounding-volume refit is involved.

h0   : the same card turned end for end (yaw + 180 deg): identical pose and footprint, the weight now in the FAR end,
       so the common-sense "slide the nearest end out" slides the light end and succeeds.
twin : the trap stays (the heavy end is the nearest end) but it points to the OPPOSITE shelf edge: the card is mirrored
       through the shelf centre along its long axis (near gap, side offset, jitter mirrored) and keeps its weight in
       the end that is now nearest. The right correction (slide the other, light end) is then over the opposite
       edge. For an x-card this makes the near end the BACKWARD one, which the task's own sampler avoids because a
       card tipping off the backward edge lands behind the shelf (agentview shows the shelf empty, not the card);
       see the arm validation (results/l3_card_shelf/arms/). For an instance whose heavy end is already the far end
       (naive succeeds there) the twin keeps the pose and turns the card end for end, so the weight is in the near
       end and the heavy edge is again the opposite one.
reveal: which shelf edge the heavy end points to, in the F2 feedback's edge words.
"""
import numpy as np

HIDDEN_KEYS = ("card_yaw", "heavy_near", "heavy_edge")

OPPOSITE = {"forward": "backward", "backward": "forward", "left": "right", "right": "left"}
GLOSS = {"forward": "away from the robot", "backward": "towards the robot", "left": "on the robot's left",
         "right": "on the robot's right"}


def _wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def h0(inst):
    if not inst["heavy_near"]:
        return {}  # already the ordinary case for the naive habit (weight away from the end that gets slid out)
    return dict(card_yaw=_wrap(inst["card_yaw"] + np.pi), heavy_near=False, heavy_edge=inst["far_edge"])


def twin(inst):
    if not inst["heavy_near"]:
        return dict(card_yaw=_wrap(inst["card_yaw"] + np.pi), heavy_near=True, heavy_edge=inst["near_edge"])
    a = 0 if inst["card_along"] == "x" else 1  # world axis of the card's long side
    c = list(inst["card_xy"])
    c[a] = 2.0 * inst["shelf_xy"][a] - c[a]
    yaw = inst["card_yaw"]
    yaw = np.pi - yaw if a == 0 else -yaw  # reflection x -> -x (resp. y -> -y) of the card's heading
    return dict(card_xy=[float(c[0]), float(c[1])], card_yaw=_wrap(yaw), near_edge=OPPOSITE[inst["near_edge"]],
                far_edge=OPPOSITE[inst["far_edge"]], heavy_edge=OPPOSITE[inst["heavy_edge"]])


def reveal(inst):
    e = inst["heavy_edge"]
    return (f"The card is not uniform: 8 g of its 10 g are a weight hidden in the end that points towards the shelf's "
            f"{e} edge ({GLOSS[e]}), so its centre of mass is 2.8 cm from the card's middle towards that end.")
