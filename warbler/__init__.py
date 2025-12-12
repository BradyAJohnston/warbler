import bpy
from bpy.utils import register_class, unregister_class
from bpy.props import PointerProperty, CollectionProperty
from bpy.app.handlers import frame_change_post

from . import ops
from . import panel
from . import props
from . import manager

CLASSES = ops.CLASSES + props.CLASSES + panel.CLASSES


def register():
    for cls in CLASSES:
        register_class(cls)
    bpy.types.Scene.SimulationManager = manager.SimulationManager()  # type: ignore
    bpy.types.Object.wb = PointerProperty(type=props.WarblerObjectProperties)  # type: ignore
    bpy.types.Scene.wb = PointerProperty(type=props.WarblerSceneProperties)  # type: ignore
    bpy.types.Scene.wb_sim_list = CollectionProperty(type=props.SimulationListItem)  # type: ignore
    frame_change_post.append(manager._step_simulations)


def unregister():
    for cls in CLASSES:
        unregister_class(cls)
    del bpy.types.Scene.SimulationManager  # type: ignore
    del bpy.types.Scene.wb_sim_list  # type: ignore
    del bpy.types.Object.wb  # type: ignore
    try:
        frame_change_post.remove(manager._step_simulations)
    except ValueError:
        pass
