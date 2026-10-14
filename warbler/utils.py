import bpy
import numpy as np
import warp as wp
from mathutils import Quaternion


def get_scene() -> bpy.types.Scene:
    """Return the current Blender scene, asserting it is not None."""
    scene = bpy.context.scene
    assert scene is not None, "bpy.context.scene is None"
    return scene


def quat_to_blender(quat: np.ndarray) -> list[float]:
    return [quat[3], quat[0], quat[1], quat[2]]


def blender_rotation(quat: Quaternion) -> np.ndarray:
    return np.array([getattr(quat, j) for j in "xyzw"])


def world_transform(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    """World-space location and rotation (xyzw quaternion) of ``obj``.

    Taken from ``matrix_world``, so parents, constraints and any rotation mode
    are accounted for. Scale is left out; it is applied to shape sizes instead.
    """
    loc, rot, _ = obj.matrix_world.decompose()
    return np.array(loc), blender_rotation(rot.normalized())


def wp_transform(obj: bpy.types.Object | None = None) -> wp.transform:
    if obj is None:
        return wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))
    loc, rot = world_transform(obj)
    return wp.transform(wp.vec3(*loc), wp.quat(*rot))


def angular_velocity(q0: np.ndarray, q1: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity rotating xyzw quaternion ``q0`` to ``q1`` in ``dt``."""
    a = Quaternion((q0[3], q0[0], q0[1], q0[2]))
    b = Quaternion((q1[3], q1[0], q1[1], q1[2]))
    delta = b @ a.inverted()
    if delta.w < 0.0:  # take the short way round
        delta.negate()
    axis, angle = delta.to_axis_angle()
    return np.array(axis) * angle / dt


def decay_lerp(a, b, decay: int, dt: float) -> np.ndarray:
    return b + (a - b) * np.exp(-decay * dt)


def smooth_lerp(a, b, decay: int = 5) -> np.ndarray:
    return b + (a - b) * (1 - np.exp(-decay * delta_t()))


def delta_t() -> float:
    scene = get_scene()
    return 1 / scene.render.fps / scene.render.fps_base
