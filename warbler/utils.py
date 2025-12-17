import numpy as np
import bpy
from mathutils import Quaternion
import warp as wp


def quat_to_blender(quat: np.ndarray) -> list[float]:
    return [quat[3], quat[0], quat[1], quat[2]]


def blender_rotation(quat: Quaternion) -> np.ndarray:
    return np.array([getattr(quat, j) for j in "xyzw"])


def wp_rotation(obj: bpy.types.Object) -> wp.quat:
    return wp.quat(*blender_rotation(obj.rotation_quaternion))


def wp_location(obj: bpy.types.Object) -> wp.vec3:
    return wp.vec3(*obj.location)


def wp_transform(obj: bpy.types.Object | None = None) -> wp.transform:
    if obj is None:
        return wp.transform(wp.vec3(0, 0, 0), wp.quat_identity(float))
    return wp.transform(wp_location(obj), wp_rotation(obj))


def decay_lerp(a, b, decay: int, dt: float) -> np.ndarray:
    return b + (a - b) * np.exp(-decay * dt)


def smooth_lerp(a, b, decay: int = 5) -> np.ndarray:
    return b + (a - b) * (1 - np.exp(-decay * delta_t()))


def delta_t():
    return 1 / bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
