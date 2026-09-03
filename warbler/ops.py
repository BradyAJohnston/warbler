from typing import ClassVar

from bpy.types import Context, Operator

from .manager import get_manager
from .simulation import SimulatorXPBD


class ReturnValues:
    RUNNING_MODAL: ClassVar[set[str]] = {"RUNNING_MODAL"}
    CANCELLED: ClassVar[set[str]] = {"CANCELLED"}
    FINISHED: ClassVar[set[str]] = {"FINISHED"}
    PASS_THROUGH: ClassVar[set[str]] = {"PASS_THROUGH"}
    INTERFACE: ClassVar[set[str]] = {"INTERFACE"}


class ReportValues:
    DEBUG: ClassVar[set[str]] = {"DEBUG"}
    INFO: ClassVar[set[str]] = {"INFO"}
    OPERATOR: ClassVar[set[str]] = {"OPERATOR"}
    PROPERTY: ClassVar[set[str]] = {"PROPERTY"}
    WARNING: ClassVar[set[str]] = {"WARNING"}
    ERROR: ClassVar[set[str]] = {"ERROR"}
    ERROR_INVALID_INPUT: ClassVar[set[str]] = {"ERROR_INVALID_INPUT"}
    ERROR_INVALID_CONTEXT: ClassVar[set[str]] = {"ERROR_INVALID_CONTEXT"}
    ERROR_OUT_OF_MEMORY: ClassVar[set[str]] = {"ERROR_OUT_OF_MEMORY"}


class BaseOperator(Operator):
    RETURN = ReturnValues
    REPORT = ReportValues

    def manager(self, context: Context):
        return get_manager(context)

    def execute(self, context: Context):
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
        except Exception as e:  # noqa: BLE001 — report any compile failure in the UI
            self.report(
                ReportValues.ERROR,  # type: ignore
                f"Unable to compile simulation, error: {e}",
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
        man.simulations.pop(item.name, None)
        man.sim_items.remove(man.item_index)
        man.item_index = max(0, man.item_index - 1)
        return ReturnValues.FINISHED


CLASSES = [WB_OT_AddSimulation, WB_OT_RemoveSimulation, WB_OT_CompileSimulation]
