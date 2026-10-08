# LFF-Bench developer guide (for everyone adding tasks)

Design doc (Chinese, not in this repo): `agentic-robotics/docs/02-benchmark设计.md` — §2 (principles) and §3 (task table).
Failure taxonomy (not in this repo): `agentic-robotics/failure_taxonomy_v0.md`. Setup: see README.md.

## Environment
```bash
cd <repo> && source run_env.sh
export OMP_NUM_THREADS=1   # 8-core machine shared by several people; keep each process single-threaded
python scripts/validate.py --task <task_name> --n 20 --k 5
```
CPU-only MuJoCo 3.2.3 + robosuite 1.4.1 + LIBERO (`third_party/LIBERO`, upstream at a pinned commit, read-only — never edit it).
Physics without rendering runs ~100 control steps/s; rendering a 256px image costs ~0.1 s. First env build ~6–20 s.
Pipe-to-`tail` buffers output until exit — for long runs write to a log file instead.

## Code layout
- `lffbench/objects.py` — procedurally generated MJCF objects (`box_xml`, `bin_xml`) registered into LIBERO's registry via `register_generated(ClassName, xml)`. Class names must be CamelCase **without digits**; the BDDL category is the snake_case of the class name. LIBERO's own objects (e.g. `cream_cheese`, `butter`, `glazed_rim_porcelain_ramekin`, `wooden_tray`, `basket`, `plate`, …) can be used directly in BDDL.
- `lffbench/bddl.py` — `write_bddl(task_name, language, objects, fixtures)`. Layout is overridden at reset, so regions only need to be valid.
- `lffbench/envs.py` — `make_env`, `render`, `obj_pos`, `place_upright(env, name, xy, yaw, surface_z)` (**use this to position LIBERO meshes** — many are modelled lying down and `set_obj_pose` with a pure yaw lays them on their side), `set_obj_pose` (raw teleport; only at reset or for injected disturbances), `set_friction`, `scale_mass`, `contacts_between`, `settle`. Table top z = `TABLE_Z` = 0.90; robot base at x = -0.66; +x points away from the robot.
- `lffbench/skills.py` — `Skills`: `move_to(xyz, yaw, speed)`, `set_gripper(close)`, `grasp_at`, `place_at`, `push`, `hold`, `holding(obj)`, `gripper_width()`. `bias` = hidden calibration error added to every commanded position. Faster OSC (kp 300, 0.1 m/step) reaches ~0.6 m/s.
- `lffbench/task_base.py` — `LFFTask` + `register_task`. Implement `make_bddl`, `sample_instance(seed)`, `apply_instance(inst)`, `outcome(inst)` (must return `success` and a human-readable `detail` for F2 feedback), and scripted references. Cross-attempt tasks implement `execute/default_params/oracle_params/adapt_params/blind_params`; within-episode tasks override `run_scripted(kind, inst, k, seed)` and must return the same history format (list of dicts with `attempt`, `params`, `outcome`).
- `lffbench/tasks/l5_slide.py` — worked example (L5, cross protocol). Copy its structure.
- `scripts/validate.py` — headroom validation; writes `results/<task>/validate.json` + thumbnails.

LIBERO's placement sampler runs inside `env.reset()` before your layout override; if it raises "Cannot place all objects", spread the BDDL regions ≥ 0.12 m apart.

## Rules when several people work in parallel
- Put everything task-specific in your own `lffbench/tasks/<file>.py`.
- Shared files (`objects.py`, `envs.py`, `skills.py`, `task_base.py`, `bddl.py`, `validate.py`): **additive edits only** (new functions / new optional arguments with backward-compatible defaults). Re-read the file right before editing. Never change the behaviour of an existing function; if you need different behaviour, subclass or copy into your task file.
- Never use `pkill -f` patterns; kill only your own PIDs.

## Acceptance criteria for a task (n ≥ 20 instances, k = 5)
| policy | target |
|---|---|
| oracle (knows hidden vars) | ≥ 90 % success at attempt 1 |
| naive (prior behaviour) | ≤ 25 % at attempt 1 |
| adaptive (uses F2 feedback) | ≥ 80 % succ@5 |
| blind (retry without history) | clearly below adaptive (aim ≤ 40 % succ@5) |
The failure of the naive policy must be **visible** in the camera images (the agent must be able to perceive it), and the information needed to fix it must be **in the outcome of the failure**, not in the initial scene alone (otherwise it is not learning from failure).
Also provide `instruction` (direct object names) and `instruction_indirect` (descriptive reference, e.g. "the yellow juice carton") — the latter tests perception / semantic reasoning.

## Deliverables per task
1. `lffbench/tasks/<file>.py` registered with `@register_task`.
2. `results/<task>/validate.json` with n ≥ 20 + thumbnails (initial scene, naive failure end-state).
3. A task card `task_cards/<task>.md`: level, capabilities, scene, hidden variables, why the prior fails, what must be learned, F2 feedback text examples, validation table, known issues / threats to validity. Separate measured facts from untested assumptions.

## Additions (2026-10-06 evening)
- **Category.** Every task sets `category` (task_base.LFFTask): one of `system_identification`, `dynamics_adaptation`,
  `failed_interaction_inference`, `state_restoration`, `procedural_memory`, `strategy_switching`, `geometry_inference`
  (user's framing: task -> hidden information -> required capability -> adapted action), plus `level` L1–L6 where it
  maps (system_identification=L1, state_restoration=L2, strategy_switching=L3, geometry_inference=L4,
  dynamics_adaptation=L5, procedural_memory=L6; failed_interaction_inference uses level "L7").
- **Agent interface will be low level.** Agents will get only `move_to(x, y, z, yaw_deg, speed)` + `open_gripper` /
  `close_gripper` (+ observation / pixel_to_world / done) — no `push` or other composite primitive. Every task must be
  solvable by a sequence of those calls; scripted policies may use helpers, but only helpers built from
  `Skills.move_to` / `Skills.set_gripper` (Skills.push is such a helper).
- **Robot-centric direction words in feedback (F2):** forward = +x (away from the robot), backward = -x, left = +y
  (robot's left), right = -y. Never image left/right. Give metric numbers in cm.
- **Write the task card early** and keep it updated (sessions can be cut off by rate limits).
- **Video:** after validation, record one demo with `python scripts/make_video.py --task <name> --kind adaptive --seed <s>`
  (pick a seed where adaptive fails first and then succeeds) and put the path in the task card.
- Existing tasks to learn from: l1_bias.py (hidden actuation bias, TrackedSkills), l2_knock.py (within protocol,
  injected disturbance), l3_wall.py (fixture wall, per-step monitor, within variant), l4_fit.py (resizing geoms per
  instance), l5_slide.py (hidden friction, fixture marker), l6_swap.py (memory task, drain_events).
