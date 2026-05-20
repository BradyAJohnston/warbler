import databpy as db
from bpy.types import Object, Context, Depsgraph
import bpy
import numpy as np

# Geometry data types that carry named attributes
_GEOM_TYPES = (bpy.types.Mesh, bpy.types.PointCloud, bpy.types.Curves)


class GeometryAttributes:
    """Wraps the evaluated data of any geometry object (Mesh, PointCloud, Curves)."""

    def __init__(
        self, data: bpy.types.Mesh | bpy.types.PointCloud | bpy.types.Curves | None
    ):
        self._data = data

    @property
    def attributes(self):
        if self._data is None:
            return None
        return self._data.attributes

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


def _as_geom_data(
    data: object,
) -> bpy.types.Mesh | bpy.types.PointCloud | bpy.types.Curves | None:
    """Narrow a data value to one of the geometry types that carry attributes."""
    if isinstance(data, (bpy.types.Mesh, bpy.types.PointCloud, bpy.types.Curves)):
        return data
    return None


class GeometrySet:
    def __init__(self, obj: Object, context: Context | None = None):
        self.obj = obj
        ctx = context if isinstance(context, Context) else bpy.context
        depsgraph: Depsgraph = ctx.evaluated_depsgraph_get()
        self.eval_obj = obj.evaluated_get(depsgraph)

        # Keep _eval_geom alive on the instance — the data blocks returned by
        # geom.pointcloud / geom.mesh are owned by this object and will be freed
        # if it goes out of scope.
        try:
            self._eval_geom = self.eval_obj.evaluated_geometry()  # type: ignore[attr-defined]
        except AttributeError:
            self._eval_geom = None

    @property
    def obj_type(self) -> str:
        return self.eval_obj.type

    @property
    def data(self) -> bpy.types.Mesh | bpy.types.PointCloud | bpy.types.Curves | None:
        """Return the best geometry data for reading particle attributes.

        Checks GN-evaluated output first (pointcloud > curves > mesh) so that a
        mesh object whose GN tree outputs points is read correctly. Falls back to
        the object's base data block for plain objects without GN modifiers.
        """
        if self._eval_geom is not None:
            for component in (
                self._eval_geom.pointcloud,  # type: ignore[attr-defined]
                self._eval_geom.curves,  # type: ignore[attr-defined]
                self._eval_geom.mesh,  # type: ignore[attr-defined]
            ):
                data = _as_geom_data(component)
                if data is not None:
                    return data

        return _as_geom_data(self.eval_obj.data)

    @property
    def pointcloud(self) -> GeometryAttributes:
        """Return attributes from whatever geometry type the source uses."""
        return GeometryAttributes(self.data)

    def _get_point_count(self) -> int:
        data = self.data
        if data is None:
            return 0
        if "position" in data.attributes:
            return len(data.attributes["position"].data)
        if isinstance(data, bpy.types.Mesh):
            return len(data.vertices)
        return 0
