import databpy as db
from bpy.types import Object, Context, Depsgraph
import bpy
import numpy as np


class GeometryAttributes:
    """Wraps the evaluated data of any geometry object (Mesh, PointCloud, Curves)."""

    def __init__(self, data):
        self._data = data

    @property
    def attributes(self):
        return self._data.attributes if self._data is not None else None

    def to_props(self) -> dict[str, np.ndarray]:
        """Return the subset of attributes warbler knows how to use as particle inputs."""
        if self._data is None:
            return {}
        attrs = self._data.attributes
        return {
            name: db.Attribute(attrs[name]).as_array()
            for name in ["position", "velocity", "mass", "radius"]
            if name in attrs
        }


class GeometrySet:
    def __init__(self, obj: Object, context: Context | None = None):
        self.obj = obj
        ctx = context if isinstance(context, Context) else bpy.context
        depsgraph: Depsgraph = ctx.evaluated_depsgraph_get()
        self.eval_obj = obj.evaluated_get(depsgraph)

    @property
    def obj_type(self) -> str:
        return self.eval_obj.type

    @property
    def data(self):
        return self.eval_obj.data

    @property
    def pointcloud(self) -> GeometryAttributes:
        """Return attributes from whatever geometry type the source uses."""
        return GeometryAttributes(self.eval_obj.data)

    def _get_point_count(self) -> int:
        data = self.eval_obj.data
        if data is None:
            return 0
        if "position" in data.attributes:
            return len(data.attributes["position"].data)
        if isinstance(data, bpy.types.Mesh):
            return len(data.vertices)
        return 0
