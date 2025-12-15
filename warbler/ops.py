from bpy.types import Operator, Context
from .manager import get_manager
from .simulation import SimulatorXPBD


class WB_OT_AddSimulation(Operator):
    bl_idname = "wb.add_simulation"
    bl_label = "New Simulation"
    bl_description = "Add a new simulation to tweak before being sent to the GPU"

    def execute(self, context):
        man = get_manager(context)
        man.add(SimulatorXPBD())
        return {"FINISHED"}


class WB_OT_CompileSimulation(Operator):
    bl_idname = "wb.compile_simulation"
    bl_label = "Compile"
    bl_description = "Compile this simulation and send to the GPU for evaluation."

    def execute(self, context):
        man = get_manager(context)
        man.active_simulation.compile()

        return {"FINISHED"}


class WB_OT_RemoveSimulation(Operator):
    bl_idname = "wb.remove_simulation"
    bl_label = "Remove Simulation"
    bl_description = (
        "Delete and remove a simulation from the list of those being computed"
    )

    def execute(self, context: Context):
        man = get_manager(context)
        item = man.active_item
        try:
            del man.simulations[item.name]
            man.sim_items.remove(man.item_index)
        except Exception:
            man.sim_items.remove(man.item_index)
        man.item_index = max(0, man.item_index - 1)
        return {"FINISHED"}


CLASSES = [WB_OT_AddSimulation, WB_OT_RemoveSimulation, WB_OT_CompileSimulation]
