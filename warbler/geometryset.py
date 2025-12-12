import databpy as db
from bpy.types import Object, Context, Depsgraph, PointCloud
import bpy
import numpy as np


class PointCloudAttributes:
    def __init__(self, pointcloud: PointCloud):
        self.pointcloud = pointcloud

    @property
    def attributes(self) -> bpy.types.AttributeGroupPointCloud:
        return self.pointcloud.attributes

    def to_props(self) -> dict[str, np.ndarray]:
        return {
            name: db.Attribute(self.attributes[name]).as_array()
            for name in ["position", "velocity", "mass", "radius"]
            if name in self.attributes
        }


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

    @property
    def pointcloud(self) -> PointCloudAttributes:
        return PointCloudAttributes(self.geom.pointcloud)

    def _get_point_count(self, attributes) -> int:
        if "position" in attributes:
            return len(db.Attribute(attributes["position"]))
        return 0
