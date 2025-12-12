from bpy.types import Operator, Context
from .manager import SimulationManager, get_manager
from .simulation import Simulation
from . import props
from bpy.props import IntProperty, BoolProperty, FloatVectorProperty


class WB_OT_StartSimulation(Operator):
    bl_idname = "wb.start_simulation"
    bl_label = "Send to GPU"
    bl_description = "Load the simulation onto the GPU, ready to be stepped on Blender's frame change"

    n_particles: IntProperty(  # type: ignore
        name="number of particles",
        description="Number of particles to initiate the simulation with",
        default=5000,
    )
    substeps: IntProperty(  # type: ignore
        name="Substeps",
        description="Number of substeps for each frame of the simulation",
        default=5,
        max=20,
        min=0,
    )
    velocity: FloatVectorProperty(  # type: ignore
        name="Initial Velocity",
        description="Initial velocity for the particles when starting the simulation",
        default=[0, 0, 10],
    )
    upvector: FloatVectorProperty(  # type: ignore
        name="Up Vector",
        description="Determins the orientation of the physics world, inputing the up vector for all calculations",
        default=[0, 0, 1],
    )

    add_ground: BoolProperty(  # type: ignore
        name="Add Ground Plane",
        description="Add a ground plane to the simulation for objects and particles to collide with",
        default=True,
    )

    def invoke(self, context, event):
        if context.area and context.area.type == "VIEW_3D":
            context.window_manager.invoke_props_dialog(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        scene = context.scene
        assert scene is not None
        manager: SimulationManager = scene.SimulationManager
        manager.add(
            Simulation(
                num_particles=self.n_particles,
                substeps=self.substeps,
                objects=[obj for obj in context.selected_objects],
                up_vector=self.upvector,
                ivelocity=self.velocity,
                add_ground=self.add_ground,
                particle_object=scene.wb.particle_source,
            )
        )
        self.report({"INFO"}, "Simulation compiled on the GPU")
        return {"FINISHED"}


class WB_OT_RemoveSimulation(Operator):
    bl_idname = "wb.remove_simulation"
    bl_label = "Remove Simulation"
    bl_description = (
        "Delete and remove a simulation from the list of those being computed"
    )

    def exectute(self, context: Context):
        scene = context.scene
        assert scene is not None

        manager = get_manager(context)
        wb = props.scene_properties(context)
        manager.remove(wb.manager_active_index)
        wb.manager_active_index = max(0, wb.manager_active_index - 1)

        return {"FINISHED"}


CLASSES = [WB_OT_StartSimulation, WB_OT_RemoveSimulation]
