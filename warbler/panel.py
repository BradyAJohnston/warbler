from bpy.types import Panel


class WB_PT_WarblerPanel(Panel):
    bl_idname = "WB_PT_WarblerPanel"
    bl_label = "Warbler"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        layout.operator("wb.start_simulation")
        layout.label(text="Simulation Settings")
        layout.prop(context.scene.wb, "rigid_decay_frames")
        obj = context.active_object
        if obj is None:
            return

        def prop(name: str):
            layout.prop(context.scene.wb, name)

        prop("simulation_substeps")
        prop("simulation_links")
        prop("spring_ke")
        prop("spring_kd")
        prop("spring_kf")
        prop("scale")
        prop("particle_radius")

        layout.separator()
        layout.label(text="Active Object Settings")
        layout.prop(obj.wb, "rigid_is_active")


CLASSES = [WB_PT_WarblerPanel]
