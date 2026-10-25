# Agent Viewer — historical replay

Export saved LFF-Bench API-player logs to one offline HTML page, including original
image bytes and the complete `rollout.mp4` when available. No model calls, simulator, server, build tools, or frontend dependencies
are required to generate or open the result.

From the repository root:

```bash
python scripts/agent_viewer/export.py \
  --run runs/gpt/exp2d/exp2d-gpt-6-astra_l5_slide_to_target_s1000_full_F2_direct_k3_1007-060153_p9738 \
  --out traces/agent-viewer-slide/index.html
```

Open the resulting HTML file in a browser. Repeat `--run` to include multiple
runs. If a run's original sandbox has moved, use `--sandbox /new/image/root`
with a single run. Relative images copied into the run directory take precedence.
The export remains usable after the original sandbox is removed.

## What the viewer shows

- Continuous execution video is the default view when `rollout.mp4` exists, with
  native playback, seeking, fullscreen and playback speed controls. The original
  MP4 is embedded byte-for-byte; no still-image slideshow is substituted.
- The separate step view retains input images and tool logs. Selecting an attempt
  opens its step details. Video and logs are not automatically synchronized because
  old recordings have no frame-to-event index and omit model waiting time.

- One step per recorded assistant response; tool calls and results stay together.
- The most recent observation **before** each decision, and separately the new
  images returned **after** its actions. Older images may also be in model context;
  the viewer shows the request's recorded image count when available.
- Camera switching, per-step playback, scrubbing, keyboard navigation (arrows and
  space), and jumps to attempt verdicts. Playback uses fixed seconds per step,
  with original elapsed/model latency shown separately.
- Tool arguments, raw tool results, public assistant replies, API retries, missing
  images, and incomplete calls. Recorded reasoning summaries are exported verbatim between observation and action in the step details.
  The summary of decision N is shown between the preceding feedback and action N.
  Public replies are labeled separately; missing summaries stay explicitly missing.
  These are recorded summaries, not full internal CoT or reconstructed reasoning.
- A reset warning when a failed `done` returns the next attempt's initial images.
  Attempt success comes from server `done` events, not the assistant's claim.

## Current input contract and limitations

Supports the API-player `ev` schema in `transcript.jsonl` or
`gpt_transcript.jsonl`, with `session.json` and optional `events.jsonl`. Other
harness formats (MCP, native Codex/Claude, RPent, GPT-Policy) need adapters and
are not claimed as supported. Corrupt JSON and ambiguous result matching fail
explicitly instead of silently pairing the wrong steps.

Image files are copied byte-for-byte at export time. Old logs contain paths rather
than image hashes, so files overwritten since the run cannot be authenticated as
the original model inputs. Missing images are visibly marked; no replacement
frames are invented. Runs without a video fall back to step replay with the video tab disabled.

The sample video is 19.05 seconds, H.264, 640×376 at 20 fps. Its action segments
were recorded at 3× simulation speed; choose ⅓× playback to slow them back down.
The recording includes both attempts and their verdict cards, but not model/API
waiting time.

The sample `l5_slide_to_target`, seed 1000, has 14 assistant responses, 13 tool
calls, and two verdicts: failure then success. It is a historical development
run using depth perception and F2 feedback, not evidence of general benchmark
performance. The first push overshot by 4 cm; the next push reduced its endpoint
from x=-0.105 m to x=-0.145 m and succeeded.

## Verification

```bash
python -m unittest discover -s tests -p test_agent_viewer.py -v
```

Regression cases cover observation/action ordering, resets, carried-forward
observations, API retries, missing artifacts, incomplete calls, reordered call
IDs, malformed logs, path restrictions, and safe embedding of untrusted text.

## Recording future runs

`api_player.py` already requests provider summaries (`auto` for OpenAI) and
saves returned `reasoning_summary` plus public `text` for each assistant turn.
`run_agent.py` now also defaults `--reasoning-summary` to `auto`; pass `none`
when an endpoint cannot support the option. Both loggers add
`reasoning_summary_status`: `recorded` when a summary is present and
`not_returned` when the response contains none. This labels response availability,
not whether summaries were requested or whether the model reasoned internally.

Historical missing records are left unchanged. No extra model calls or
retrospective rationale generation are used. Changes apply to newly started
processes from this checkout; other worktrees and native CLI harnesses retain
their own logging behavior. The viewer still imports API-player transcripts;
`run_agent` logs use a different schema and need an adapter to view here.

## Live terminal viewer

`live.py` observes existing runs without contacting the robot/model or changing the
run. Use a terminal with truecolor support for low-resolution color image previews;
`--ascii` provides a monochrome fallback. Pillow is the only image dependency and
is already present in the evaluation environment.

```bash
python scripts/agent_viewer/live.py --batch /absolute/path/to/runs/api/BATCH
python scripts/agent_viewer/live.py --run /absolute/path/to/RUN --once --ascii
```

A batch view follows the first running run and switches when the manifest advances;
for concurrent runs in one batch, use `--run` to select another. Refresh defaults to
6 seconds. `Ctrl+C` exits; `--no-images`, `--width`, `--image-rows`, `--interval`, and
`--duration` control presentation. A redirected output uses ASCII and no screen
control codes. `--once` prints one snapshot.

Supported live sources: API-player transcripts; framework gateway/MCP tool logs;
and Codex `item.completed` events. Only successful `view_images` attachments or
API-player image inputs become displayed observations. Robot-produced files alone
do not establish that the Agent saw them. Matching embedded images from native
Codex events take precedence over file references; potentially overwritten files
are marked unavailable. The current view displays the latest observation batch,
not continuous video, and explicitly labels its age. Recorded summaries are shown
when the relevant stream provides them. It does not expose `session.json` tokens,
server-side hidden task parameters, or send any robot commands.

```bash
python -m unittest discover -s tests -p test_live_viewer.py -v
```

### Clearer terminal images

The old character preview was limited to 20 rows (at most ~40 vertical pixels).
It now uses available terminal height and width. `--camera main` or `--camera wrist`
selects a larger single view. Character mode is still a low-resolution preview.

For actual image pixels, use `--renderer iterm` (iTerm2 / WezTerm / VS Code with
`terminal.integrated.enableImages: true`) or `--renderer kitty`. Native mode
encodes original-resolution pixels as PNG; the terminal fits the display without
reducing the transmitted image to text-cell resolution. `auto` recognizes iTerm2,
WezTerm and Kitty environment markers; unknown terminals and multiplexers retain
the ANSI fallback. VS Code needs an explicit `--renderer iterm` after its setting
is enabled. Native protocol encoding is tested; visual compatibility still needs
to be checked in the user's terminal. Use a direct terminal outside tmux/screen
if image passthrough has not been configured. Redirected output always stays text.

Existing CLI processes keep their original interval and renderer: exit with Ctrl+C
and restart to pick up changes. Evaluation processes need no restart.

The user's Claude development terminal (`TERM_PROGRAM=claude-desktop`) displayed
an empty image area with the iTerm transport. In that environment, even an explicit
`--renderer iterm` now falls back to ANSI with a visible explanation. This avoids
blank frames but does not provide native image detail. Use `--camera main` for a
larger single-camera preview; original-resolution viewing needs an image-capable
viewer. No Claude terminal setting enabling inline images has been verified.
