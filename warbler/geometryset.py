import bpy
import databpy as db
import numpy as np
from bpy.types import Context, Depsgraph, Object

# Geometry data types that carry named attributes
_GEOM_TYPES = (bpy.types.Mesh, bpy.types.PointCloud, bpy.types.Curves)

# POINT-domain attributes warbler reads from every geometry component
PARTICLE_ATTRIBUTES = (
    "position",
    "velocity",
    "mass",
    "radius",
    "pinned",
    "on_surface",
    "surface_offset",
    "attach",
)


class GeometryAttributes:
    """Wraps the evaluated data of any geometry object (Mesh, PointCloud, Curves)."""

    def __init__(
        self,
        data: bpy.types.Mesh | bpy.types.PointCloud | bpy.types.Curves | None,
        owner: "GeometrySet | None" = None,
    ):
        self._data = data
        # Evaluated data is freed with the GeometrySet that produced it, so
        # keep it alive for as long as these attributes are.
        self._owner = owner

    @property
    def attributes(self):
        if self._data is None:
            return None
        return self._data.attributes

    def to_props(self) -> dict[str, np.ndarray]:
        """Return the subset of POINT attributes warbler uses as particle inputs."""
        if self._data is None:
            return {}
        attrs = self._data.attributes
        return {
            name: db.Attribute(attrs[name]).as_array()
            for name in PARTICLE_ATTRIBUTES
            if name in attrs and attrs[name].domain == "POINT"
        }

    def __len__(self) -> int:
        if self._data is None or "position" not in self._data.attributes:
            return 0
        return len(self._data.attributes["position"].data)  # ty: ignore[unresolved-attribute]

    @property
    def curve_sizes(self) -> np.ndarray:
        """Number of points in each curve (Curves data only)."""
        if not isinstance(self._data, bpy.types.Curves):
            return np.zeros(0, dtype=np.int32)
        offsets = np.zeros(len(self._data.curve_offset_data), dtype=np.int32)
        self._data.curve_offset_data.foreach_get("value", offsets)
        return np.diff(offsets)

    @property
    def triangles(self) -> np.ndarray:
        """(N, 3) vertex indices of the mesh's triangulated faces (Mesh data only)."""
        if not isinstance(self._data, bpy.types.Mesh) or len(self._data.polygons) == 0:
            return np.zeros((0, 3), dtype=np.int32)
        self._data.calc_loop_triangles()
        tris = np.zeros(len(self._data.loop_triangles) * 3, dtype=np.int32)
        self._data.loop_triangles.foreach_get("vertices", tris)
        return tris.reshape(-1, 3)

    @property
    def edges(self) -> np.ndarray:
        """(N, 2) vertex indices of the mesh's edges (Mesh data only)."""
        if not isinstance(self._data, bpy.types.Mesh) or len(self._data.edges) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        edges = np.zeros(len(self._data.edges) * 2, dtype=np.int32)
        self._data.edges.foreach_get("vertices", edges)
        return edges.reshape(-1, 2)

    def edge_attribute(self, name: str) -> np.ndarray | None:
        """An EDGE-domain attribute of the mesh, if it has one."""
        if not isinstance(self._data, bpy.types.Mesh):
            return None
        attr = self._data.attributes.get(name)
        if attr is None or attr.domain != "EDGE":
            return None
        return db.Attribute(attr).as_array()

    def point_attributes(self) -> dict[str, tuple[np.ndarray, str]]:
        """Every user POINT attribute (except position) with its data type.

        Used to carry source attributes (e.g. Molecular Nodes' ``res_id``,
        ``chain_id``) over to the simulation output.
        """
        if self._data is None:
            return {}
        out = {}
        for attr in self._data.attributes:
            if (
                attr.domain != "POINT"
                or attr.name == "position"
                or attr.name.startswith(".")
            ):
                continue
            out[attr.name] = (db.Attribute(attr).as_array(), attr.data_type)
        return out

    @property
    def faces(self) -> list[list[int]]:
        """Polygon vertex indices of the mesh, for rebuilding it as an output."""
        if not isinstance(self._data, bpy.types.Mesh):
            return []
        return [list(p.vertices) for p in self._data.polygons]


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
        return GeometryAttributes(self.data, self)

    def _component(self, name: str, kind: type) -> GeometryAttributes:
        """Attributes of one component of the evaluated geometry, if present.

        Without a GN-evaluated geometry set, the object's own data is used when
        it matches ``kind``.
        """
        if self._eval_geom is not None:
            data = getattr(self._eval_geom, name, None)
        else:
            data = self.eval_obj.data
        return GeometryAttributes(
            _as_geom_data(data) if isinstance(data, kind) else None, self
        )

    @property
    def points(self) -> GeometryAttributes:
        """The point cloud component — simulated as free particles."""
        return self._component("pointcloud", bpy.types.PointCloud)

    @property
    def curves(self) -> GeometryAttributes:
        """The curves component — simulated as spring chains (ropes / hair)."""
        return self._component("curves", bpy.types.Curves)

    @property
    def mesh(self) -> GeometryAttributes:
        """The mesh component — faces are simulated as cloth."""
        return self._component("mesh", bpy.types.Mesh)

    def _get_point_count(self) -> int:
        data = self.data
        if data is None:
            return 0
        if "position" in data.attributes:
            return len(data.attributes["position"].data)  # ty: ignore[unresolved-attribute]
        if isinstance(data, bpy.types.Mesh):
            return len(data.vertices)
        return 0
