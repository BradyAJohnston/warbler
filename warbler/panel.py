from bpy.types import Panel
from bpy.types import UILayout
from .ops import WB_OT_AddSimulation, WB_OT_RemoveSimulation, WB_OT_CompileSimulation
from .props import scene_properties
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
        layout.prop(context.scene.wb, "rigid_decay_frames")
        obj = context.active_object
        if obj is None:
            return

        layout.prop(context.scene.render, "fps")

        layout.separator()
        layout.label(text="Active Object Settings")
        layout.prop(obj.wb, "rigid_is_active")

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
        col.operator(WB_OT_AddSimulation.bl_idname)
        col.operator(WB_OT_RemoveSimulation.bl_idname)
        item_collection = context.scene.wb_sim_list
        if len(item_collection) == 0:
            return
        try:
            item = man.sim_items[man.item_index]
        except KeyError:
            return
        if item is None:
            return

        def prop(name: str):
            try:
                layout.prop(item, name)
            except Exception as e:
                print(e)

        prop("simulation_substeps")
        prop("simulation_links")
        prop("spring_ke")
        prop("spring_kd")
        prop("spring_kf")
        prop("scale")
        prop("particle_radius")
        prop(name="particle_source")
        layout.prop(item, "is_active")
        row = layout.row()
        row.scale_y = 2
        row.operator(WB_OT_CompileSimulation.bl_idname)


CLASSES = [WB_PT_WarblerPanel]
