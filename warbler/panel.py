from bpy.types import Panel
from bpy.types import UILayout


class WB_PT_WarblerPanel(Panel):
    bl_idname = "WB_PT_WarblerPanel"
    bl_label = "Warbler"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout: UILayout = self.layout
        assert layout is not None and context is not None
        layout.operator("wb.start_simulation")
        layout.label(text="Simulation Settings")
        layout.prop(context.scene.wb, "rigid_decay_frames")
        obj = context.active_object
        if obj is None:
            return

        layout.prop(context.scene.render, "fps")

        def prop(name: str):
            layout.prop(context.scene.wb, name)

        prop("simulation_substeps")
        prop("simulation_links")
        prop("spring_ke")
        prop("spring_kd")
        prop("spring_kf")
        prop("scale")
        prop("particle_radius")
        prop(name="particle_source")

        layout.separator()
        layout.label(text="Active Object Settings")
        layout.prop(obj.wb, "rigid_is_active")

        layout.template_list(
            "WB_UL_SimulationList",
            "A list",
            context.scene,
            "wb_sim_list",
            context.scene.wb,
            "manager_active_index",
            rows=3,
        )
        layout.operator("wb.remove_simulation")
        item = context.scene.wb_sim_items.get(context.scene.wb.manager_active_index)
        if item is None:
            return
        layout.prop(item, "is_active")


CLASSES = [WB_PT_WarblerPanel]
