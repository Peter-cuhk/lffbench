# Running LFF-Bench with your own model API

This page is for running the benchmark on your own machine with your own API key. It covers setup, a smoke test,
the main runs, and what to send back. Background on the tasks: `README.md` (Chinese) and `task_cards/<task>.md`.

**What a run is.** A VLM agent controls a simulated Franka arm (LIBERO / MuJoCo) through three tools: `robot`
(`move_eef x y z yaw gripper speed`, `observe`, `finish`, …), `shell` (an isolated sandbox with python, numpy,
OpenCV; no network) and `view_images`. It sees RGB images only. After each attempt it is told only
"Attempt succeeded." or "Attempt failed." The question is whether it learns from failure, either within an
attempt or across attempts. The driver is `scripts/api_player.py` (one run). `scripts/api_batch.py` schedules many
runs. Every model gets the same system prompt (the task README), the same tools and the same budgets; only the
wire format differs per provider.

## 1. Requirements
- Linux x86_64 (the agent's isolated shell uses Linux user namespaces; macOS cannot run it).
- root once, to install the isolated shell under `/opt/lffjail`.
- `git`, [`uv`](https://docs.astral.sh/uv/), OSMesa for CPU rendering (`sudo apt install libosmesa6`). No GPU needed.
- ~2.5 GB disk. CPU: every concurrent run uses one simulator process. Plan about one CPU core per concurrent run.

## 2. Install
```bash
git clone https://github.com/Peter-cuhk/lffbench.git && cd lffbench
bash setup/install.sh            # .venv (Python 3.10) + third_party/LIBERO (pinned commit) + .libero/config.yaml
sudo bash jail/setup_jail.sh     # isolated shell for the agent (no network, cannot see /home /mnt /tmp)
cp .env.example .env             # then fill in the key(s) for your provider, see below
source run_env.sh                # in every new shell
python scripts/check_setup.py    # packages, simulation + rendering, isolated shell: all lines must say [ok]
```
On Ubuntu >= 23.10, if the shell check fails with `uid_map` / `Operation not permitted`, run
`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`. More troubleshooting is in `README.md` §2.

## 3. API keys (`.env`)
| `--provider` | API | variables |
|---|---|---|
| `anthropic` | Anthropic Messages | `ANTHROPIC_API_KEY` (optional `ANTHROPIC_BASE_URL`) |
| `openai` | OpenAI Responses | `OPENAI_API_KEY` (optional `OPENAI_BASE_URL`) |
| `gemini` | Google Gemini | `GEMINI_API_KEY` |
| `chat` | any OpenAI-compatible Chat Completions endpoint (vLLM, SGLang, xAI, gateways) | `LFF_CHAT_BASE_URL`, `LFF_CHAT_API_KEY` |

`--effort` (default `high`) is mapped to each provider's reasoning setting. Use the provider's exact model name for
`--model`.

**Gateways and relays.** If your key goes through a gateway that routes one model name to several upstream channels,
check each channel before the main run: some relays (e.g. ones built on coding-assistant subscriptions) silently add
their own system prompt, which changes the model under test. Send a one-line prompt such as "Reply with the word
pong." and look at the reported input tokens: about 10–20 is clean; hundreds or thousands means a hidden prompt was
added. Pin a clean channel for the whole run (many gateways accept the channel id as `--model`), and report which one
you used. Prompt caching through such gateways is often ineffective; that only affects cost.

## 4. Smoke test (one cheap run, about 5–15 min)
```bash
python scripts/api_batch.py start --batch smoke --provider anthropic --model <model-name> \
    --tasks l1_height_bias:3000 --k 1 --memfree upfront --conds noharness --indep-reps 1 --concurrency 1
python scripts/api_batch.py status --batch smoke     # wait until the run is "done"
python scripts/api_batch.py table  --batch smoke
```
Logs go to `runs/api/smoke/<run>/`: `events.jsonl` is the server log (commands, judged outcome), `transcript.jsonl` is
the model conversation including token usage. Check the token counts to estimate your cost per run before the main
batch. Anthropic and Gemini backends were verified offline against the real SDKs but have had little live use. Please
report anything odd.

## 5. Main runs
The main task set is `sets/main_v1.txt` (task:seeds per line; 24 tasks, 92 instances). The protocol is the
`api_batch.py` default:
- per instance, 1 run with history (`harness`, one agent with up to 3 attempts, `k=3`);
- history runs do not reveal k up front (`--hide-k`, default): their README says one attempt, exactly like a fresh
  single-attempt agent, and more attempts are announced only after a failure;
- memory-free retries (`--memfree followup`, default since 2026-10-19): attempt 1 of the history run also counts as
  the first memory-free attempt, since under `--hide-k` it sees exactly what a fresh agent sees. Only if it fails does
  the runner add a fresh single-attempt agent (`noharness`, `k=1`, no memory of the failure) for attempt 2, and if
  that fails too, another one for attempt 3. The first success ends the chain. `table` reports these as
  "no-harness @1 / @k" (the memory-free pass@k). The older protocol (3 independent runs per instance, all before the
  history runs) is `--memfree upfront`;
- feedback F1 (success / failure only), RGB only.

```bash
python scripts/api_batch.py start --batch main_<model> --provider <provider> --model <model-name> --effort high \
    --tasks $(grep -v '^#' sets/main_v1.txt) --k 3 --concurrency <≈ CPU cores> --video
python scripts/api_batch.py status --batch main_<model>
python scripts/api_batch.py retry  --batch main_<model>   # re-run failed / stopped / timed-out runs (rate limits etc.)
python scripts/api_batch.py table  --batch main_<model>
```
This is 92 history runs per model, plus up to 2 memory-free runs for each instance whose first attempt fails (at most
92 × 3 = 276 runs). `--seeds-per-task 1` keeps only the first seed of each task (24 instances) for a cheaper pass. If
you started a batch before 2026-10-19, `python scripts/api_batch.py followup --batch <batch>` switches it to this
protocol; `retry` then starts the runner. A history run of a long-horizon task can need 300+ model calls; at 20–60 s
per call that is 2–5 h of wall time. A run is killed after `--run-timeout` seconds (default 8 h) and marked timed out;
raise it if your endpoint is slow. To start smaller, run a subset first (pass fewer `--tasks`), check cost, then continue with a new
`--batch` name. `python scripts/api_batch.py contract --batch a --batch b` checks that two batches used the same task
code and protocol and can share one table.

**Task changes since the first release.** 2026-10-16: `l7_key_turn` no longer limits the number of turns per
attempt. The instruction no longer says "only one turn is allowed"; an attempt succeeds once the lock is open, and
turning the key the wrong way first is logged in the outcome as `trap_fired` / `recovered_in_attempt`. If you ran
`l7_key_turn` before this change, re-run those instances: `api_batch.py contract` flags them, because the task code
hash in their contract differs.

2026-10-19: `l5_slide_to_target` and `l7_offcenter_push` no longer limit the number of strikes or pushes per attempt,
so a miss can be corrected within the same attempt. In `l5_slide_to_target` the cube must end at rest inside the
square, and its last motion must be a free slide of at least 3 cm after it leaves the gripper (pushing it slowly all
the way, or placing it, does not count); a strike now returns only after the cube has stopped. Later the same day its
far side was relaxed: the cube may lie on the far green border line but not stick out beyond it (judged: centre at
most 2 cm past the square's centre; the near side and the sides stay at 1 cm), because a cube up to about 1 cm past
that line cannot be told apart in the agentview. The instruction says so. In `l7_offcenter_push` every push must be
made with the closed gripper. Its instruction now states only the end state (it no longer says "straight forward" or
"without turning"), since turning the box and then correcting it within the attempt is allowed. Missed strikes and
bad pushes are logged as `trap_fired` / `recovered_in_attempt`. Re-run runs of these two tasks made before this
change. `l5_tablecloth_pull` keeps its one-pull rule. Also on 2026-10-19, for all tasks, the upper limit of the arm's
elbow joint (`robot0_joint4`) was tightened from -0.07 to -0.55 rad (`lffbench/envs.py`): a fast move out of a far,
stretched pose could lock the elbow, after which every later move stopped short for the rest of the attempt (44 of
714 logged attempts). Reach and the scripted reference policies are unchanged, but every task's contract hash
changed, so runs made before 2026-10-19 cannot share a table with later ones; `api_batch.py contract` flags them.

Please do not modify task code, the README template or the tools. If you must change something to get it running,
tell us exactly what you changed.

## 6. What to send back
```bash
tar czf lff_<model>.tgz --exclude='*.mp4' runs/api/<batch>/      # add the .mp4 files too if size allows
python scripts/api_batch.py table --batch <batch> > table_<model>.md
```
Also send the git commit hash (`git rev-parse HEAD`) and the exact model name / effort you used.
