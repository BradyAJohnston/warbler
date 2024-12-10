from bpy.types import PropertyGroup
from bpy.props import (
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
    BoolProperty,
)


class WarblerObjectPeroperties(PropertyGroup):
    rigid_active: BoolProperty(  # type: ignore
        name="Is Active",
        description="Active ridid body in the simulation, updating it's position based on forces",
        default=False,
    )


CLASSES = [WarblerObjectPeroperties]
