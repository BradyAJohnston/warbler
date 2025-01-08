import numpy as np
import bpy


def decay_lerp(a, b, decay: int, dt: float) -> np.ndarray:
    return b + (a - b) * np.exp(-decay * dt)


def smooth_lerp(a, b, decay: int = 5) -> np.ndarray:
    return b + (a - b) * (1 - np.exp(-decay * delta_t()))


def delta_t():
    return 1 / bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
