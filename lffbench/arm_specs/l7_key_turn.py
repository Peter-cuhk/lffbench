"""Controlled-arm spec for l7_key_turn (2026-10-12).

Hidden variable: inst["direction"], the way the key opens the lock ("CCW" or "CW", seen from above). apply_instance()
derives the joint range and outcome()/check_unlock()/oracle_params() derive the open sign from it on every reset, so
overriding "direction" alone is enough (nothing is cached at sampling time).

The scripted prior (naive) turns counterclockwise, so the trap exists only on CW locks -- all full5 instances are CW.

h0     -> direction = "CCW": same lock position / yaw, the key opens the common-sense way, naive succeeds.
twin   -> {} (no valid twin). The task has a single binary hidden variable and the trap exists only for one of its two
          values: a twin with the trap still active must be CW again, i.e. need exactly the same correction ("turn
          clockwise"), and the sign-flipped twin (CCW) is the H0 twin, where naive succeeds. There is no second hidden
          constraint (the required angle is the fixed quarter turn named in the instruction). The mismatched-history
          arm therefore has to take another instance's history -- note that every trap instance's history teaches the
          same correction, so for this task a mismatched history is not misleading about the direction.
reveal -> states which way the key turns and that the other way stops after ~3 deg (the words "clockwise",
          "counterclockwise", "seen from above" are those of the task's F2 feedback).
"""

HIDDEN_KEYS = ("direction",)


def h0(inst):
    return {"direction": "CCW"}


def twin(inst):
    return {}


def reveal(inst):
    ok, other = (("counterclockwise", "clockwise") if inst["direction"] == "CCW" else ("clockwise", "counterclockwise"))
    return (f"Note: this lock opens only when the key is turned {ok} (seen from above); turned {other} the key moves "
            f"about 3 degrees and then hits a hard stop.")
