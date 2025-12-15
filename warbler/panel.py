from bpy.types import Panel
from bpy.types import UILayout
from .ops import WB_OT_AddSimulation, WB_OT_RemoveSimulation, WB_OT_CompileSimulation
from .manager import get_manager


class WB_PT_WarblerPanel(Panel):
    bl_idname = "WB_PT_WarblerPanel"
    bl_label = "Warbler"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout: UILayout = self.layout
        assert layout is not None and context is not None
        man = get_manager(context)
        layout.label(text="Simulation Settings")
        obj = context.active_object
        if obj is None:
            return

        layout.prop(context.scene.render, "fps")

        layout.separator()
        layout.label(text="Active Object Settings")
        # layout.prop(obj.wb, "rigid_is_active")

        row = layout.row()

        row.template_list(
            "WB_UL_SimulationList",
            "A list",
            context.scene,
            "wb_sim_list",
            context.scene.wb,
            "manager_active_index",
            rows=3,
        )

        col = row.column()
        col.operator(WB_OT_AddSimulation.bl_idname, text="", icon="ADD")
        col.operator(WB_OT_RemoveSimulation.bl_idname, text="", icon="REMOVE")

        try:
            item = man.active_item
        except Exception as e:
            print(e)
            return

        col = layout.column()
        col.enabled = not item.is_compiled

        if item.is_compiled:
            full_time = item.time_sync + item.time_compute
            col.label(text=f"Simulation time: {full_time * 1e3:,.2f} ms")
            col.label(text=f"Compute:  {item.time_compute * 1e3:,.2f} ms")
            col.label(text=f"Sync:  {item.time_sync * 1e3:,.2f} ms")

        col.prop(item, "rigid_decay_frames")
        col.prop(item, "spring_ke")
        col.prop(item, "spring_kd")
        col.prop(item, "spring_kf")
        col.prop(item, "scale")
        col.prop(item, "particle_radius")
        col.prop(item, "particle_source")
        col.prop(item, "substeps")
        col.prop(item, "is_active")
        row = col.row()
        row.scale_y = 2
        row.operator(WB_OT_CompileSimulation.bl_idname)


CLASSES = [WB_PT_WarblerPanel]
