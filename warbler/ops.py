from bpy.props import IntProperty
from bpy.types import Operator, Context
from .manager import get_manager
from .simulation import SimulatorXPBD
from . import props


class WB_OT_AddSimulation(Operator):
    bl_idname = "wb.add_simulation"
    bl_label = "New Simulation"
    bl_description = "Add a new simulation to tweak before being sent to the GPU"

    def invoke(self, context, event):
        if context.area and context.area.type == "VIEW_3D":
            context.window_manager.invoke_props_dialog(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        scene = context.scene
        assert scene is not None
        man = get_manager(context)
        man.add(SimulatorXPBD())
        self.report({"INFO"}, "Simulation compiled on the GPU")
        return {"FINISHED"}


class WB_OT_CompileSimulation(Operator):
    bl_idname = "wb.compile_simulation"
    bl_label = "Compile"
    bl_description = "Compile this simulation and send to the GPU for evaluation."

    sim_item_index: IntProperty()  # type: ignore

    def execute(self, context):
        man = get_manager(context)
        sim = man.get(man.item_index)
        sim.compile()

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


CLASSES = [WB_OT_AddSimulation, WB_OT_RemoveSimulation, WB_OT_CompileSimulation]
