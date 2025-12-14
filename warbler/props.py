from bpy.types import PropertyGroup, Object, Context
import bpy
from bpy.props import (
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
    BoolProperty,
    PointerProperty,
    FloatVectorProperty,
)


class SimulationListItem(bpy.types.PropertyGroup):
    name: StringProperty(name="UUID")  # type: ignore
    compute_time: FloatProperty(name="Time", default=0.0)  # type: ignore
    is_active: BoolProperty(name="Active", default=True)  # type: ignore
    is_compiled: BoolProperty(name="Compiled", default=False)  # type: ignore
    substeps: IntProperty(name="Substeps", default=5, min=0, soft_max=100)  # type: ignore
    device: EnumProperty(  # type: ignore
        items=(
            ("cuda", "CUDA", "Compile for the GPU and simulate using CUDA"),
            (
                "cpu",
                "CPI",
                "Compile for running on the CPU and not using CUDA on the GPU",
            ),
        ),
        default="cuda",
    )

    use_ground_plane: BoolProperty("Ground Plane", default=True)  # type: ignore
    ground_plane_vector: FloatVectorProperty("Ground Normal", default=(0, 0, 1))  # type: ignore

    spring_ke: FloatProperty(  # type: ignore
        name="Spring Stiffness",
        description="Stiffness constant for springs in the simulation",
        default=1.0e5,
    )
    spring_kd: FloatProperty(  # type: ignore
        name="Spring Damping",
        description="Damping constant for springs in the simulation",
        default=250.0,
    )
    spring_kf: FloatProperty(  # type: ignore
        name="Spring Friction",
        description="Friction constant for springs in the simulation",
        default=500.0,
    )
    scale: FloatProperty(  # type: ignore
        name="Simulation Scale",
        description="Scale factor for the simulation, scaling all distances and sizes",
        default=1.0,
    )
    particle_radius: FloatProperty(  # type: ignore
        name="Particle Radius",
        description="Radius of the particles in the simulation",
        default=0.1,
    )
    particle_source: PointerProperty(  # type: ignore
        type=Object,  # type: ignore
        name="Particle Source",
        description="Source simulation particles from this object",
    )
    particle_source_evaluate: BoolProperty(  # type: ignore
        name="Evaluate",
        description="Evaluate the object and all modifiers first before sourcing the particles",
        default=True,
    )
    rigid_decay_frames: IntProperty(  # type: ignore
        name="Rigid Input Smoothing",
        description="Number of frames to move the inputs of the rigid body over",
        default=5,
        min=1,
    )


class WB_UL_SimulationList(bpy.types.UIList):
    def draw_item(  # type: ignore
        self,
        context,
        layout: bpy.types.UILayout,
        data,
        item: SimulationListItem,
        icon,
        active_data,
        active_property,
        *,
        index=0,
        flt_flag=0,
    ):
        layout: bpy.types.UILayout = layout

        layout.label(text=item.name)

        layout.prop(item, "is_active", text="")


class WarblerSceneProperties(PropertyGroup):
    manager_active_index: IntProperty()  # type: ignore

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


def scene_properties(context: Context | None) -> WarblerSceneProperties:
    if context is None:
        context = bpy.context

    return context.scene.wb  # type: ignore


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
    rigid_density: FloatProperty(  # type: ignore
        name="Density",
        description="Density of the rigid body, used to compute mass from volume",
        default=1000.0,
    )
    rigid_ka: FloatProperty(  # type: ignore
        name="Contact Adhesion",
        description="Adhesion constant for contacts involving this rigid body",
        default=0.0,
    )
    rigid_kd: FloatProperty(  # type: ignore
        name="Contact Damping",
        description="Damping constant for contacts involving this rigid body",
        default=1000.0,
    )
    rigid_ke: FloatProperty(  # type: ignore
        name="Contact Elasticity",
        description="Elasticity constant for contacts involving this rigid body",
        default=1.0e5,
    )
    rigid_kf: FloatProperty(  # type: ignore
        name="Friction damping",
        description="The friction damping coefficient. Used by SemiImplcit, Feathertone",
        default=500.0,
    )
    rigid_mu: FloatProperty(  # type: ignore
        name="Friction Coefficient",
        description="Coulomb friction coefficient for contacts, used by all solvers",
        default=0.5,
    )
    rigid_restitution: FloatProperty(  # type: ignore
        name="Restitution Coefficient",
        description="Restitution (bounciness) coefficient for contacts, used by XPBD",
        default=0.0,
    )
    rigid_thickness: FloatProperty(  # type: ignore
        name="Collision Thickness",
        description="The thickness of the shape used for collision detection",
        default=1e-5,
    )


CLASSES = [
    WarblerObjectProperties,
    WarblerSceneProperties,
    SimulationListItem,
    WB_UL_SimulationList,
]
