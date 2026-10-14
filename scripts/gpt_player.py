"""Old entry point of the GPT player, kept so existing commands keep working: the code now lives in api_player.py
(the unified harness for every provider). This runs it with --provider openai and the old transcript name.

python scripts/gpt_player.py --session runs/claude/<run>/session.json --model gpt-6-astra --effort medium [--style la]
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_player import main  # noqa: E402

if __name__ == "__main__":
    try:
        main(default_provider="openai", transcript_name="gpt_transcript.jsonl")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
