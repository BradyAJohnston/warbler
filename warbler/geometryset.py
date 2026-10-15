from typing import NamedTuple

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
    "affinity",
    "affinity_type",
)

# Points flagged ``interaction`` are rows of the interaction table, not
# particles: a pair of types with the depth and range of their attraction
INTERACTION_FLAG = "interaction"
INTERACTION_ATTRIBUTES = (
    "interaction_a",
    "interaction_b",
    "interaction_depth",
    "interaction_range",
)


class GeometryAttributes:
    """Wraps the evaluated data of any geometry object (Mesh, PointCloud, Curves)."""

    def __init__(
        self,
        data: bpy.types.Mesh | bpy.types.PointCloud | bpy.types.Curves | None,
        owner: "GeometrySet | None" = None,
        rows: np.ndarray | None = None,
    ):
        self._data = data
        # Evaluated data is freed with the GeometrySet that produced it, so
        # keep it alive for as long as these attributes are.
        self._owner = owner
        # Boolean mask of the points that belong here (None: all of them)
        self._rows = rows

    def _array(self, attr) -> np.ndarray:
        values = db.Attribute(attr).as_array()
        return values if self._rows is None else values[self._rows]

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
            name: self._array(attrs[name])
            for name in PARTICLE_ATTRIBUTES
            if name in attrs and attrs[name].domain == "POINT"
        }

    def __len__(self) -> int:
        if self._data is None or "position" not in self._data.attributes:
            return 0
        if self._rows is not None:
            return int(self._rows.sum())
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
                or attr.name == INTERACTION_FLAG
                or attr.name in INTERACTION_ATTRIBUTES
            ):
                continue
            out[attr.name] = (self._array(attr), attr.data_type)
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

    def _interaction_rows(self) -> np.ndarray | None:
        """Which points of the point cloud are interaction-table rows."""
        data = self._component("pointcloud", bpy.types.PointCloud)._data
        if data is None or INTERACTION_FLAG not in data.attributes:
            return None
        return db.Attribute(data.attributes[INTERACTION_FLAG]).as_array().astype(bool)

    @property
    def points(self) -> GeometryAttributes:
        """The point cloud component — simulated as free particles, except the
        points that are interaction-table rows."""
        points = self._component("pointcloud", bpy.types.PointCloud)
        table = self._interaction_rows()
        if table is not None:
            points._rows = ~table
        return points

    @property
    def interactions(self) -> dict[str, np.ndarray]:
        """The interaction table: a pair of interaction types per row
        (``interaction_a`` / ``_b``) and their attraction's ``interaction_depth``
        and ``interaction_range``. Empty when the source has none."""
        table = self._interaction_rows()
        if table is None or not table.any():
            return {}
        rows = GeometryAttributes(
            self._component("pointcloud", bpy.types.PointCloud)._data, self, table
        )
        attrs = rows.attributes
        assert attrs is not None
        return {
            name: rows._array(attrs[name])
            for name in INTERACTION_ATTRIBUTES
            if name in attrs
        }

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

    @property
    def instances(self) -> "InstanceAttributes":
        """The top-level instances — simulated as rigid bodies."""
        if self._eval_geom is None:
            return InstanceAttributes(None, [], self)
        return InstanceAttributes(
            self._eval_geom.instances_pointcloud(),  # type: ignore[attr-defined]
            list(self._eval_geom.instance_references()),  # type: ignore[attr-defined]
            self,
        )


class ReferenceMesh(NamedTuple):
    """An instanced mesh: vertices, (M, 3) triangles, and the POINT ``binding``
    strength and ``binding_type`` of each vertex (binding sites), if it has
    them."""

    vertices: np.ndarray
    triangles: np.ndarray
    binding: np.ndarray | None
    binding_type: np.ndarray | None = None
    normals: np.ndarray | None = None


# INSTANCE-domain attributes warbler reads for rigid bodies
INSTANCE_ATTRIBUTES = ("collider", "velocity", "density")


def _vertex_normals(mesh: bpy.types.Mesh) -> np.ndarray:
    normals = np.zeros(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertex_normals.foreach_get("vector", normals)
    return normals.reshape(-1, 3)


def _point_attribute(mesh: bpy.types.Mesh, name: str) -> np.ndarray | None:
    attr = mesh.attributes.get(name)
    if attr is None or attr.domain != "POINT":
        return None
    return db.Attribute(attr).as_array()


def _reference_mesh(reference: object) -> bpy.types.Mesh | None:
    """The mesh of one instance reference (a geometry set, object or collection)."""
    if isinstance(reference, bpy.types.Object):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        reference = reference.evaluated_get(depsgraph).evaluated_geometry()  # type: ignore[attr-defined]
    data = getattr(reference, "mesh", None)
    return data if isinstance(data, bpy.types.Mesh) else None


class InstanceAttributes:
    """The instances of an evaluated geometry: one transform per instance, each
    pointing (``reference_index``) at one of a list of referenced geometries.
    """

    def __init__(
        self,
        pointcloud: bpy.types.PointCloud | None,
        references: list,
        owner: GeometrySet,
    ):
        self._data = pointcloud
        self._references = references
        self._owner = owner

    def __len__(self) -> int:
        if self._data is None:
            return 0
        return len(self._data.points)

    @property
    def transforms(self) -> np.ndarray:
        """(N, 4, 4) instance matrices, translation in the last column."""
        if self._data is None:
            return np.zeros((0, 4, 4), dtype=np.float32)
        attr = self._data.attributes["instance_transform"]
        return db.Attribute(attr).as_array().transpose(0, 2, 1)

    @property
    def reference_index(self) -> np.ndarray:
        if self._data is None:
            return np.zeros(0, dtype=np.int32)
        return db.Attribute(self._data.attributes[".reference_index"]).as_array()

    def to_props(self) -> dict[str, np.ndarray]:
        """The subset of INSTANCE attributes warbler uses as rigid body inputs."""
        if self._data is None:
            return {}
        attrs = self._data.attributes
        return {
            name: db.Attribute(attrs[name]).as_array()
            for name in INSTANCE_ATTRIBUTES
            if name in attrs
        }

    def reference_geometry(self) -> list[ReferenceMesh | None]:
        """The mesh of each reference, None if it has none."""
        out: list[ReferenceMesh | None] = []
        for reference in self._references:
            mesh = _reference_mesh(reference)
            if mesh is None or len(mesh.vertices) == 0:
                out.append(None)
                continue
            vertices = np.zeros(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", vertices)
            out.append(
                ReferenceMesh(
                    vertices.reshape(-1, 3),
                    GeometryAttributes(mesh).triangles,
                    _point_attribute(mesh, "binding"),
                    _point_attribute(mesh, "binding_type"),
                    _vertex_normals(mesh),
                )
            )
        return out
