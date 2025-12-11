import databpy as db
from bpy.types import Object, Context, Depsgraph
import bpy


class GeometrySet:
    def __init__(self, obj: Object, context: None | Context = None):
        self.obj = obj
        self.context = context if isinstance(context, Context) else bpy.context
        depsgraph: Depsgraph = self.context.view_layer.depsgraph
        if depsgraph is None:
            raise ValueError

        self.eval_obj = depsgraph.id_eval_get(self.obj)
        self.geom = self.eval_obj.evaluated_geometry()  # type: ignore

    @property
    def instances(self):
        return self.geom.instances_pointcloud()

    def _get_point_count(self, attributes) -> int:
        if "position" in attributes:
            return len(db.Attribute(attributes["position"]))
        return 0
