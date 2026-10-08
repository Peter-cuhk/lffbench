"""Procedurally generated objects for LFF-Bench, registered into LIBERO's object registry.

Each generated object is an MJCF file with the layout robosuite's MujocoXMLObject expects
(an outer body containing body "object" plus bottom/top/horizontal_radius sites).
"""
import os
import re

import numpy as np
from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import OBJECTS_DICT, register_object

GEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets_gen")
os.makedirs(GEN_DIR, exist_ok=True)

_SOLID = 'solimp="0.998 0.998 0.001" solref="0.001 1"'


def _fmt(v):
    return " ".join(f"{x:.5f}" for x in v)


def _wrap(model_name, geoms_xml, half_height, radius):
    return f"""<mujoco model="{model_name}">
  <worldbody>
    <body>
      <body name="object">
{geoms_xml}
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {-half_height:.5f}" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {half_height:.5f}" name="top_site" />
      <site rgba="0 0 0 0" size="0.005" pos="{radius:.5f} {radius:.5f} 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>
"""


def box_xml(model_name, half, rgba, density=500.0, friction=(1.0, 0.005, 0.0001)):
    """Solid box; one geom used for both collision and rendering."""
    g = (f'        <geom name="{model_name}_g0" type="box" size="{_fmt(half)}" rgba="{_fmt(rgba)}" '
         f'density="{density}" friction="{_fmt(friction)}" {_SOLID} group="0" />\n'
         f'        <geom type="box" size="{_fmt(half)}" rgba="{_fmt(rgba)}" conaffinity="0" contype="0" group="1" />')
    return _wrap(model_name, g, half[2], float(np.hypot(half[0], half[1])))


def bin_xml(model_name, inner_half_xy, wall_h, wall_t=0.006, floor_t=0.006, rgba=(0.6, 0.6, 0.65, 1.0),
            density=800.0):
    """Open-top box. Origin at the center of the outer footprint, z=0 at mid height."""
    ix, iy = inner_half_xy
    H = (wall_h + floor_t) / 2.0
    geoms = []
    parts = [
        ((0, 0, -H + floor_t / 2), (ix + wall_t, iy + wall_t, floor_t / 2)),  # floor
        ((ix + wall_t / 2, 0, floor_t / 2), (wall_t / 2, iy + wall_t, wall_h / 2)),  # +x wall
        ((-ix - wall_t / 2, 0, floor_t / 2), (wall_t / 2, iy + wall_t, wall_h / 2)),  # -x wall
        ((0, iy + wall_t / 2, floor_t / 2), (ix, wall_t / 2, wall_h / 2)),  # +y wall
        ((0, -iy - wall_t / 2, floor_t / 2), (ix, wall_t / 2, wall_h / 2)),  # -y wall
    ]
    for k, (p, s) in enumerate(parts):
        geoms.append(f'        <geom name="{model_name}_g{k}" type="box" pos="{_fmt(p)}" size="{_fmt(s)}" '
                     f'rgba="{_fmt(rgba)}" density="{density}" friction="1.0 0.005 0.0001" {_SOLID} group="0" />')
        geoms.append(f'        <geom type="box" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(rgba)}" '
                     f'conaffinity="0" contype="0" group="1" />')
    return _wrap(model_name, "\n".join(geoms), H, float(np.hypot(ix + wall_t, iy + wall_t)))


def _key(class_name):
    return "_".join(re.sub(r"([A-Z0-9])", r" \1", class_name).split()).lower()


def register_generated(class_name, xml_text, free=True):
    """Write the MJCF and register a LIBERO object class. Returns the BDDL category name."""
    key = _key(class_name)
    path = os.path.join(GEN_DIR, f"{key}.xml")
    with open(path, "w") as f:
        f.write(xml_text)
    if key in OBJECTS_DICT:
        return key

    default_joints = [dict(type="free", damping="0.0005")] if free else None

    def __init__(self, name=key, joints="default"):
        j = default_joints if joints == "default" else joints
        MujocoXMLObject.__init__(self, path, name=name, joints=j, obj_type="all", duplicate_collision_geoms=False)
        self.category_name = key
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}

    cls = type(class_name, (MujocoXMLObject,), {"__init__": __init__})
    register_object(cls)
    return key
