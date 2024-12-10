from bpy.types import Operator
from .manager import SimulationManager
from .warbler import Warbler
from bpy.props import IntProperty, BoolProperty


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
        manager.simulation = Warbler(
            num_particles=self.n_particles, substeps=self.substeps, links=self.springs
        )
        self.report({"INFO"}, "Simulation compiled on the GPU")
        return {"FINISHED"}


CLASSES = [WB_OT_StartSimulation]
