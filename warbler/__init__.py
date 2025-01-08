import bpy
from bpy.utils import register_class, unregister_class
from bpy.props import PointerProperty
from bpy.app.handlers import frame_change_post

from . import ops
from . import panel
from . import props
from .manager import SimulationManager, _step_simulations

CLASSES = ops.CLASSES + props.CLASSES + panel.CLASSES


def register():
    for cls in CLASSES:
        register_class(cls)
    bpy.types.Scene.SimulationManager = SimulationManager()
    bpy.types.Object.wb = PointerProperty(type=props.WarblerObjectProperties)
    bpy.types.Scene.wb = PointerProperty(type=props.WarblerSceneProperties)
    frame_change_post.append(_step_simulations)


def unregister():
    for cls in CLASSES:
        unregister_class(cls)
    del bpy.types.Scene.SimulationManager
    del bpy.types.Object.wb
    try:
        frame_change_post.remove(_step_simulations)
    except ValueError:
        pass
