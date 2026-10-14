"""Per-task specs for the controlled arms of full5 (2026-10-12, docs/14 §6).

One module per task, named exactly like the task (lffbench/arm_specs/<task_name>.py), defining:

  HIDDEN_KEYS : tuple of instance keys that carry the hidden variable(s)
  h0(inst)    -> dict   overrides for the H0 twin: same layout, hidden variable at its "ordinary" value, so the
                        common-sense first attempt succeeds (scripted naive succeeds)
  twin(inst)  -> dict   overrides for the mismatched-history source: same layout, the trap is still active (scripted
                        naive still fails, visibly) but the right correction is different from the original instance's,
                        ideally opposite (e.g. centre of mass on the other side)
  reveal(inst)-> str    H+ text appended to the task instruction: the hidden variable stated plainly in robot-frame
                        words a careful human could act on, without prescribing the solution

All three take the ORIGINAL sampled instance and return only the keys to change (JSON-serialisable). The server
applies them with dict.update() after sample_instance(); reset_instance() must honour them.
"""
import importlib


def load(task_name):
    """The spec module for task_name, or None if the task has no controlled-arm spec."""
    try:
        return importlib.import_module(f"{__name__}.{task_name}")
    except ModuleNotFoundError as e:
        if e.name == f"{__name__}.{task_name}":
            return None
        raise


def overrides(task_name, arm, inst):
    """Instance overrides for arm in (h0, twin); {} for other arms."""
    if arm not in ("h0", "twin"):
        return {}
    spec = load(task_name)
    if spec is None:
        raise SystemExit(f"--arm {arm}: no spec in lffbench/arm_specs/{task_name}.py")
    return getattr(spec, arm)(inst)


def reveal(task_name, inst):
    spec = load(task_name)
    if spec is None:
        raise SystemExit(f"--arm hplus: no spec in lffbench/arm_specs/{task_name}.py")
    return spec.reveal(inst)
