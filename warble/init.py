import bpy

CLASSES = []


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in CLASSES:
        bpy.utils.unregister_class(cls)
