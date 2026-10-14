"""Standalone fake model API (no real model) for offline batch tests of scripts/api_batch.py --harness ...

  python tests/harness_selfcheck/fake_server.py --api anthropic|responses --port 15690 --script oracle|text --for claude|codex|ref
Runs until killed; records nothing unless --record <dir>. The "oracle" script is selfcheck.ORACLE (l1_height_bias seed
3000) translated for the harness that will call this server (--for).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from lffbench.harnesses.fakeapi import FakeAnthropic, FakeResponses  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True, choices=["anthropic", "responses"])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--script", default="oracle", choices=["oracle", "text"])
    ap.add_argument("--for", dest="target", default="claude", choices=["claude", "codex", "ref"])
    ap.add_argument("--record", default=None)
    a = ap.parse_args()
    import selfcheck
    script = selfcheck.translate(selfcheck.ORACLE, a.target) if a.script == "oracle" else []
    srv = (FakeAnthropic if a.api == "anthropic" else FakeResponses)(script, record_dir=a.record, port=a.port).start()
    print(f"fake {a.api} on {srv.base_url}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
