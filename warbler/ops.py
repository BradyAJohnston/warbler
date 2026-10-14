from typing import ClassVar

from bpy.props import EnumProperty
from bpy.types import Context, Operator

from . import demos
from .manager import get_manager
from .simulation import Simulator


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
        man.add(Simulator())
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


class WB_OT_FreeSimulation(BaseOperator):
    bl_idname = "wb.free_simulation"
    bl_label = "Free"
    bl_description = (
        "Stop the simulation and release it from the device, keeping its output "
        "objects. Solver, device and iterations can then be changed before "
        "compiling again"
    )

    def execute(self, context):
        man = self.manager(context)
        item = man.active_item
        sim = man.simulations.get(item.name)
        if sim is not None:
            sim.free()
        else:
            # e.g. after reloading a file: the compiled state didn't survive
            item.is_compiled = False
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


class WB_OT_AddDemo(BaseOperator):
    bl_idname = "wb.add_demo"
    bl_label = "Add Demo"
    bl_description = (
        "Add a demo simulation whose source geometry is built with Geometry Nodes"
    )
    bl_options = {"REGISTER", "UNDO"}  # noqa: RUF012

    demo: EnumProperty(  # type: ignore
        name="Demo",
        items=[(key, name, desc) for key, (name, desc, _) in demos.DEMOS.items()],
    )

    def execute(self, context: Context):
        _, _, build = demos.DEMOS[self.demo]
        sim = build(context)
        try:
            sim.compile()
        except Exception as e:  # noqa: BLE001 — report any compile failure in the UI
            self.report(ReportValues.ERROR, f"Unable to compile demo, error: {e}")  # type: ignore
            return ReturnValues.CANCELLED
        return ReturnValues.FINISHED


CLASSES = [
    WB_OT_AddSimulation,
    WB_OT_RemoveSimulation,
    WB_OT_CompileSimulation,
    WB_OT_FreeSimulation,
    WB_OT_AddDemo,
]
