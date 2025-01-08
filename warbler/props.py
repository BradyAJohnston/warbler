from bpy.types import PropertyGroup
from bpy.props import (
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
    BoolProperty,
)


class WarblerSceneProperties(PropertyGroup):
    rigid_decay_frames: IntProperty(  # type: ignore
        name="Rigid Input Smoothing",
        description="Number of frames to move the inputs of the rigid body over",
        default=5,
        min=1,
    )
    simulation_substeps: IntProperty(  # type: ignore
        name="Simulation Substeps",
        description="Number of substeps per frame",
        default=5,
    )
    simulation_links: BoolProperty(  # type: ignore
        name="Simulation Links",
        description="Enable links between objects in the simulation",
        default=False,
    )


class WarblerObjectProperties(PropertyGroup):
    uuid: StringProperty(  # type: ignore
        name="UUID",
        description="Unique identifier for this object, for linking to objects in the simulation",
        default="",
    )
    rigid_body_index: IntProperty(  # type: ignore
        name="Rigid Body Index",
        description="Index of the rigid body in the simulation",
        default=-1,
    )
    rigid_is_active: BoolProperty(  # type: ignore
        name="Is Active",
        description="Active ridid body in the simulation, updating it's position based on forces",
        default=False,
    )
    rigid_shape: EnumProperty(  # type: ignore
        name="Shape",
        description="Shape of the rigid body in the simulation",
        items=[
            ("CUBE", "Cube", "Cube shape for the rigid body"),
            ("SPHERE", "Sphere", "Sphere shape for the rigid body"),
            ("PLANE", "Plane", "Plane shape for the rigid body"),
            ("MESH", "Mesh", "Uses the triangular mesh for the object for collisions"),
            ("CONE", "Cone", "Cone shape for the rigid body"),
            ("CYLINDER", "Cylinder", "Cylinder shape for the rigid body"),
            ("CAPSULE", "Capsule", "Capsule shape for the rigid body"),
        ],
        default="CUBE",
    )


CLASSES = [WarblerObjectProperties, WarblerSceneProperties]
