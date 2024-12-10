from bpy.types import Operator


class WB_OT_StartSimulation(Operator):
    bl_idname = "wb.start_simulation"
    bl_label = "Start Simulation"
    bl_description = "Load the simulation onto the GPU, ready to be stepped on Blender's frame change"

    def execute(self, context):
        return super().execute(context)


CLASSES = [WB_OT_StartSimulation]
