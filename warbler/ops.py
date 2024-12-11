from bpy.types import Operator
from .manager import SimulationManager
from .simulation import Simulation
from bpy.props import IntProperty, BoolProperty, FloatVectorProperty


class WB_OT_StartSimulation(Operator):
    bl_idname = "wb.start_simulation"
    bl_label = "Start Simulation"
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
    springs: BoolProperty(  # type: ignore
        name="Springs",
        description="Create springs between succesive points for linking together in the simulation",
        default=False,
    )

    def invoke(self, context, event):
        if context.area and context.area.type == "VIEW_3D":
            context.window_manager.invoke_props_dialog(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        manager: SimulationManager = context.scene.SimulationManager
        manager.simulation = Simulation(
            num_particles=self.n_particles,
            substeps=self.substeps,
            links=self.springs,
            objects=[obj for obj in context.selected_objects],
            up_vector=self.upvector,
            ivelocity=self.velocity,
        )
        self.report({"INFO"}, "Simulation compiled on the GPU")
        return {"FINISHED"}


CLASSES = [WB_OT_StartSimulation]
