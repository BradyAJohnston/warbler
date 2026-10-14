import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Context, Object, PropertyGroup


class SimulationListItem(bpy.types.PropertyGroup):
    name: StringProperty(name="UUID")  # type: ignore
    time_compute: FloatProperty(name="Time", default=0.0)  # type: ignore
    time_sync: FloatProperty(name="Time", default=0.0)  # type: ignore
    is_active: BoolProperty(name="Active", default=True)  # type: ignore
    is_compiled: BoolProperty(name="Compiled", default=False)  # type: ignore
    substeps: IntProperty(name="Substeps", default=10, min=1, soft_max=100)  # type: ignore
    iterations: IntProperty(  # type: ignore
        name="Solver Iterations",
        description=(
            "XPBD constraint-solve iterations per substep. Higher values make "
            "springs/contacts stiffer and more convergent (Newton examples use 10). "
            "Independent of substeps, which subdivide the frame timestep"
        ),
        default=10,
        min=1,
        soft_max=50,
    )
    solver: EnumProperty(  # type: ignore
        name="Solver",
        items=(
            (
                "XPBD",
                "XPBD",
                "Position-based dynamics. Supports particle-particle collisions; best for particles and curves",
            ),
            (
                "VBD",
                "VBD",
                "Vertex block descent. Proper membrane and bending energies; best for cloth. No particle-particle collisions",
            ),
        ),
        default="XPBD",
    )
    device: EnumProperty(  # type: ignore
        items=(
            ("cpu", "CPU", "Simulate on the CPU (safe default; no shared GPU context)"),
            (
                "cuda",
                "CUDA",
                "Simulate on the GPU via CUDA (faster, but a crash poisons the Blender CUDA context until restart)",
            ),
        ),
        default="cuda",
    )
    use_ground_plane: BoolProperty("Ground Plane", default=True)  # type: ignore
    ground_plane_vector: FloatVectorProperty("Ground Normal", default=(0, 0, 1))  # type: ignore

    sim_rigid_collection: PointerProperty(  # type: ignore
        type=bpy.types.Collection,
        name="Collection",
        description="Collection of objects that will be used as rigid bodies in the simulation.",
    )

    spring_ke: FloatProperty(  # type: ignore
        name="Spring Stiffness",
        description=(
            "Stretch stiffness of springs along curves (and along cloth edges "
            "with XPBD)"
        ),
        default=1.0e3,
        min=0.0,
    )
    spring_kd: FloatProperty(  # type: ignore
        name="Spring Damping",
        description="Damping of curve and cloth springs",
        default=1.0,
        min=0.0,
    )
    spring_kf: FloatProperty(  # type: ignore
        name="Spring Friction",
        description="Friction constant for springs in the simulation",
        default=500.0,
    )
    bend_ke: FloatProperty(  # type: ignore
        name="Bend Stiffness",
        description=(
            "Stiffness of the springs that skip a point along each curve, "
            "resisting folding. 0 disables them"
        ),
        default=1.0e2,
        min=0.0,
    )
    cloth_tri_ke: FloatProperty(  # type: ignore
        name="Stretch Stiffness",
        description="Cloth membrane stretch stiffness (VBD)",
        default=1.0e3,
        min=0.0,
    )
    cloth_tri_ka: FloatProperty(  # type: ignore
        name="Area Stiffness",
        description="Cloth membrane area-preservation stiffness (VBD)",
        default=1.0e3,
        min=0.0,
    )
    cloth_tri_kd: FloatProperty(  # type: ignore
        name="Stretch Damping",
        description="Cloth membrane damping (VBD)",
        default=1.0e-1,
        min=0.0,
    )
    cloth_edge_ke: FloatProperty(  # type: ignore
        name="Bend Stiffness",
        description="Cloth bending stiffness across mesh edges",
        default=1.0e1,
        min=0.0,
    )
    cloth_edge_kd: FloatProperty(  # type: ignore
        name="Bend Damping",
        description="Cloth bending damping across mesh edges",
        default=0.0,
        min=0.0,
    )
    scale: FloatProperty(  # type: ignore
        name="Simulation Scale",
        description="Scale factor for the simulation, scaling all distances and sizes",
        default=1.0,
    )
    source: PointerProperty(  # type: ignore
        type=Object,
        name="Source",
        description=(
            "Object whose evaluated geometry is simulated: point clouds become "
            "particles, curves become spring chains and meshes with faces become cloth"
        ),
    )
    source_evaluate: BoolProperty(  # type: ignore
        name="Evaluate",
        description="Evaluate the object and all modifiers first before sourcing the particles",
        default=True,
    )
    soft_contact_ke: FloatProperty(  # type: ignore
        name="Contact Stiffness",
        description="Stiffness of particle soft contacts (particles vs ground/rigid bodies)",
        default=1e2,
    )
    soft_contact_kd: FloatProperty(  # type: ignore
        name="Contact Damping",
        description="Damping of particle soft contacts",
        default=1.0,
    )
    soft_contact_mu: FloatProperty(  # type: ignore
        name="Contact Friction",
        description="Friction coefficient for particle soft contacts",
        default=0.5,
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
        layout.label(text=item.name)

        layout.prop(item, "is_active", text="")


class WarblerSceneProperties(PropertyGroup):
    manager_active_index: IntProperty()  # type: ignore
    sim_list: CollectionProperty(type=SimulationListItem)  # type: ignore


def scene_properties(context: Context | None) -> WarblerSceneProperties:
    if context is None:
        context = bpy.context

    return context.scene.wb  # type: ignore


def object_properties(obj: Object) -> "WarblerObjectProperties":
    """Typed accessor for the `wb` property group registered on Object."""
    return obj.wb  # type: ignore


class WarblerObjectProperties(PropertyGroup):
    uuid: StringProperty(  # type: ignore
        name="UUID",
        description="Unique identifier for this object, for linking to objects in the simulation",
        default="",
    )
    mass: FloatProperty(name="Mass", description="Mass of the rigid body", default=1e2)  # type: ignore
    sim_body_index: IntProperty(  # type: ignore
        name="Rigid Body Index",
        description="Index of the rigid body in the simulation",
        default=-1,
    )
    is_active: BoolProperty(  # type: ignore
        name="Is Active",
        description="Active ridid body in the simulation, updating it's position based on forces",
        default=False,
    )
    sim_shape: EnumProperty(  # type: ignore
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
    rigid_margin: FloatProperty(  # type: ignore
        name="Collision Margin",
        description="Contact margin around the shape used for collision detection",
        default=0.0,
    )


CLASSES = [
    # have to make sure we register the item -> list -> collection in order
    # otherwise Blender won't know about the others when registering
    SimulationListItem,
    WB_UL_SimulationList,
    WarblerObjectProperties,
    WarblerSceneProperties,
]
