"""Controlled-arm spec for l2_turntable_restore (2026-10-12).

Hidden variable: inst["q0"], the turntable's start angle (rad, + = counter-clockwise seen from above). It is the only
thing the agent has to remember: the main task forces a large turn of the disc (to bring the butter out from under the
shelf), after which nothing in the scene shows the start angle. apply_instance() places the disc at q0 and puts the
butter at disc-frame angle theta (so its world azimuth is theta + q0) on every reset, and reset_instance() records the
start angle from the simulator, so overriding "q0" alone is enough.

Geometry used below (TurntableRestore.butter_to_basket): with the butter at world azimuth az (about the disc axis)
and box yaw az + tilt, the main task turns the disc by T = wrap(PICK_AZ - tilt - az), PICK_AZ = +90 deg (open side).
Its sign flips at az = flip = -90 deg - tilt (the middle of the covered side); |T| = 180 deg - |az - flip|.

h0     -> q0 with the butter already at PICK_AZ (open side, long axis along world y): the main task needs no turn
          (T ~ 0), so the scripted naive (main task, then stop) leaves the disc at its start angle and succeeds.
          NB: the instruction's "under the shelf" is then not literally true (the butter is on the open half).
twin   -> q0 with the butter still under the shelf but 35 deg on the OTHER side of flip: the main task turns the disc
          the opposite way (|T| = 145 deg, inside the measured 136-179 deg range), so naive still fails visibly, and the
          correction (turn back) has the opposite sign; the twin's start angle differs from the original's by
          35 deg + |az - flip| >= 35 deg (tolerance 15 deg), so restoring to the twin's remembered marks fails here.
reveal -> the start angle stated as the direction the yellow stripe points (seen from above), in robot-centric words
          (forward = away from the robot, counter-clockwise as in the F2 feedback).
"""
import numpy as np

HIDDEN_KEYS = ("q0",)

TARGET = "butter_1"
PICK_AZ = np.pi / 2  # TurntableRestore.PICK_AZ
TWIN_OFF = np.radians(35.0)
STRIPE_ANG = np.pi  # disc-frame angle of the yellow stripe (l2_turntable.STRIPE_ANG)


def _wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _geom(inst):
    th, tilt = float(inst["theta"][TARGET]), float(inst["tilt"][TARGET])
    return th, tilt, -np.pi / 2 - tilt


def h0(inst):
    th, tilt, _ = _geom(inst)
    return {"q0": _wrap(PICK_AZ - tilt - th)}


def twin(inst):
    th, tilt, flip = _geom(inst)
    s = 1.0 if _wrap(th + float(inst["q0"]) - flip) >= 0 else -1.0
    return {"q0": _wrap(flip - s * TWIN_OFF - th)}


def _compass(deg):
    names = ("forward", "forward-left", "left", "backward-left", "backward", "backward-right", "right", "forward-right")
    return names[int(np.floor(((deg % 360) + 22.5) / 45.0)) % 8]


def reveal(inst):
    az = np.degrees(_wrap(float(inst["q0"]) + STRIPE_ANG))
    side = "counter-clockwise" if az >= 0 else "clockwise"
    return (f"Note: at the start, seen from above, the yellow stripe on the turntable points out from the turntable's "
            f"centre {abs(az):.0f} deg {side} from straight forward (roughly {_compass(az)}); that is the angle the "
            f"turntable is at now.")
