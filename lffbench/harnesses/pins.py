"""Pinned launch fingerprints (fail-closed launch, docs/15 §0.4).

A pin says: "harness <name> at version <v> (binary sha256 <b>) with model <m> and effort <e> sends exactly this preset
prompt and tool list" (the normalised fingerprint of the first request, see claude_code.request_fingerprint /
codex.request_fingerprint). Pins are written only by tests/harness_selfcheck/selfcheck.py --write-pins after every
offline check passed for that adapter; a launch without a matching pin is refused.

Only hashes and sizes are stored here (the repo is public); the full fingerprint of each launch is in the run's
harness_preflight.json.
"""
import json
import os
import re
import time

PINS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pins")


def key(adapter, version, binary_sha, model, effort):
    return f"{adapter}|{version}|{(binary_sha or 'nobin')[:16]}|{model}|{effort}"


def _path(k, pins_dir=None):
    return os.path.join(pins_dir or PINS_DIR, re.sub(r"[^A-Za-z0-9.-]+", "_", k) + ".json")


def load(k, pins_dir=None):
    p = _path(k, pins_dir)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def write(k, fingerprint_sha256, summary, pins_dir=None):
    os.makedirs(pins_dir or PINS_DIR, exist_ok=True)
    d = dict(key=k, fingerprint_sha256=fingerprint_sha256, written=time.strftime("%Y-%m-%d %H:%M:%S"), summary=summary)
    with open(_path(k, pins_dir), "w") as f:
        json.dump(d, f, indent=1)
    return d
