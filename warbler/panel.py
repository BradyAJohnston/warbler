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
        layout.prop(context.active_object.wb, "rigid_is_active")


CLASSES = [WB_PT_WarblerPanel]
