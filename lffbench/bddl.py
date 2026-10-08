"""Tiny BDDL writer for tabletop problems. Layout is overridden at reset by the task, so the
regions here only need to be valid placement areas."""
import os

from .objects import GEN_DIR


def region(name, x0, y0, x1, y1, yaw=None):
    s = f"""      ({name}
          (:target main_table)
          (:ranges (
              ({x0} {y0} {x1} {y1})
            )
          )"""
    if yaw is not None:
        s += f"""
          (:yaw_rotation (
              ({yaw} {yaw})
            )
          )"""
    return s + "\n      )\n"


def write_bddl(task_name, language, objects, fixtures=(), extra_regions="", goal="(And)"):
    """objects / fixtures: list of (instance_name, category, region_name, (x0,y0,x1,y1))."""
    regs = ""
    for inst, cat, rname, box in list(objects) + list(fixtures):
        regs += region(rname, *box)
    regs += extra_regions
    fx = "\n".join(f"    {i} - {c}" for i, c, _, _ in fixtures)
    ob = "\n".join(f"    {i} - {c}" for i, c, _, _ in objects)
    init = "\n".join(f"    (On {i} main_table_{r})" for i, _, r, _ in list(objects) + list(fixtures))
    text = f"""(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {language})
    (:regions
{regs}    )

  (:fixtures
    main_table - table
{fx}
  )

  (:objects
{ob}
  )

  (:obj_of_interest
  )

  (:init
{init}
  )

  (:goal
    {goal}
  )

)
"""
    path = os.path.join(GEN_DIR, f"{task_name}.bddl")
    with open(path, "w") as f:
        f.write(text)
    return path
