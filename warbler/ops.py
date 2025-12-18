from bpy.types import Context, Operator

from .manager import get_manager
from .simulation import SimulatorXPBD


class ReturnValues:
    RUNNING_MODAL = {"RUNNING_MODAL"}
    CANCELLED = {"CANCELLED"}
    FINISHED = {"FINISHED"}
    PASS_THROUGH = {"PASS_THROUGH"}
    INTERFACE = {"INTERFACE"}


class ReportValues:
    DEBUG = {"DEBUG"}
    INFO = {"INFO"}
    OPERATOR = {"OPERATOR"}
    PROPERTY = {"PROPERTY"}
    WARNING = {"WARNING"}
    ERROR = {"ERROR"}
    ERROR_INVALID_INPUT = {"ERROR_INVALID_INPUT"}
    ERROR_INVALID_CONTEXT = {"ERROR_INVALID_CONTEXT"}
    ERROR_OUT_OF_MEMORY = {"ERROR_OUT_OF_MEMORY"}


class BaseOperator(Operator):
    RETURN = ReturnValues
    REPORT = ReportValues

    def manager(self, context: Context):
        return get_manager(context)

    def execute(self, context: Context):  # type: ignore
        return self.RETURN.FINISHED


class WB_OT_AddSimulation(BaseOperator):
    bl_idname = "wb.add_simulation"
    bl_label = "New Simulation"
    bl_description = "Add a new simulation to tweak before being sent to the GPU"

    def execute(self, context):
        man = self.manager(context)
        man.add(SimulatorXPBD())
        return ReturnValues.FINISHED


class WB_OT_CompileSimulation(BaseOperator):
    bl_idname = "wb.compile_simulation"
    bl_label = "Compile"
    bl_description = "Compile this simulation and send to the GPU for evaluation."

    def execute(self, context):
        man = self.manager(context)
        try:
            man.active_simulation.compile()
        except Exception as e:
            self.report(
                ReportValues.ERROR,
                "Unable to compile simulation, error: {}".format(e),
            )
        return ReturnValues.FINISHED


class WB_OT_RemoveSimulation(BaseOperator):
    bl_idname = "wb.remove_simulation"
    bl_label = "Remove Simulation"
    bl_description = (
        "Delete and remove a simulation from the list of those being computed"
    )

    def execute(self, context: Context):
        man = self.manager(context)
        item = man.active_item
        try:
            del man.simulations[item.name]
            man.sim_items.remove(man.item_index)
        except Exception:
            man.sim_items.remove(man.item_index)
        man.item_index = max(0, man.item_index - 1)
        return ReturnValues.FINISHED


CLASSES = [WB_OT_AddSimulation, WB_OT_RemoveSimulation, WB_OT_CompileSimulation]
