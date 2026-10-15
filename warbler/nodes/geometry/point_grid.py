# Node-group asset "Point Grid" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
# _build_group() is the source of truth: the docstring, __init__ and accessors are regenerated from it on the next dump.
from typing import TYPE_CHECKING

from bpy.types import GeometryNodeTree
from nodebpy import TreeBuilder
from nodebpy import geometry as g
from nodebpy.builder import (
    AssetGeometryGroup,
    BooleanSocket,
    FloatSocket,
    GeometrySocket,
    IntegerSocket,
    PackageLibrary,
    SocketAccessor,
    VectorSocket,
)
from nodebpy.types import InputBoolean, InputFloat, InputInteger, InputVector


class PointGrid(AssetGeometryGroup):
    """
    A jittered 3D grid of points, as Newton's add_particle_grid

    Parameters
    ----------
    count_x : InputInteger
        Count X
    count_y : InputInteger
        Count Y
    count_z : InputInteger
        Count Z
    spacing : InputFloat
        Distance between points
    origin : InputVector
        The first point, or the grid's centre in X and Y with Centre
    centre : InputBoolean
        Centre the grid on Origin in X and Y
    jitter : InputFloat
        Largest random offset of each point along each axis
    seed : InputInteger
        Seed

    Inputs
    ------
    i.count_x : IntegerSocket
        Count X
    i.count_y : IntegerSocket
        Count Y
    i.count_z : IntegerSocket
        Count Z
    i.spacing : FloatSocket
        Distance between points
    i.origin : VectorSocket
        The first point, or the grid's centre in X and Y with Centre
    i.centre : BooleanSocket
        Centre the grid on Origin in X and Y
    i.jitter : FloatSocket
        Largest random offset of each point along each axis
    i.seed : IntegerSocket
        Seed

    Outputs
    -------
    o.points : GeometrySocket
        Points
    """

    _name = "Point Grid"
    _asset_name = "Point Grid"
    _library = PackageLibrary(__file__, "../../assets/nodes.blend")
    _color_tag = "GEOMETRY"
    _tree_properties = {
        "description": "A jittered 3D grid of points, as Newton's add_particle_grid"
    }

    class _Inputs(SocketAccessor):
        count_x: IntegerSocket
        """Count X"""
        count_y: IntegerSocket
        """Count Y"""
        count_z: IntegerSocket
        """Count Z"""
        spacing: FloatSocket
        """Distance between points"""
        origin: VectorSocket
        """The first point, or the grid's centre in X and Y with Centre"""
        centre: BooleanSocket
        """Centre the grid on Origin in X and Y"""
        jitter: FloatSocket
        """Largest random offset of each point along each axis"""
        seed: IntegerSocket
        """Seed"""

    class _Outputs(SocketAccessor):
        points: GeometrySocket
        """Points"""

    if TYPE_CHECKING:

        @property
        def i(self) -> _Inputs: ...
        @property
        def o(self) -> _Outputs: ...

    def __init__(
        self,
        count_x: InputInteger = 10,
        count_y: InputInteger = 10,
        count_z: InputInteger = 10,
        spacing: InputFloat = 0.1,
        origin: InputVector = None,
        centre: InputBoolean = False,
        jitter: InputFloat = 0.0,
        seed: InputInteger = 0,
    ):
        super().__init__(
            **{
                "Count X": count_x,
                "Count Y": count_y,
                "Count Z": count_z,
                "Spacing": spacing,
                "Origin": origin,
                "Centre": centre,
                "Jitter": jitter,
                "Seed": seed,
            }
        )

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        count_x = tree.inputs.integer("Count X", 10, min_value=1)
        count_y = tree.inputs.integer("Count Y", 10, min_value=1)
        count_z = tree.inputs.integer("Count Z", 10, min_value=1)
        spacing = tree.inputs.float(
            "Spacing", 0.1, description="Distance between points", min_value=0.0
        )
        origin = tree.inputs.vector(
            "Origin",
            (0.0, 0.0, 0.0),
            description="The first point, or the grid's centre in X and Y with Centre",
        )
        centre = tree.inputs.boolean(
            "Centre", False, description="Centre the grid on Origin in X and Y"
        )
        jitter = tree.inputs.float(
            "Jitter",
            0.0,
            description="Largest random offset of each point along each axis",
            min_value=0.0,
        )
        seed = tree.inputs.integer("Seed", 0)
        points = tree.outputs.geometry("Points")

        with g.Frame("Grid cell"):
            index = g.Index()
            combine_xyz = g.CombineXYZ(
                x=index.o.index % count_x,
                y=index.o.index // count_x % count_y,
                z=index.o.index // (count_x * count_y),
            )
            _string = g.String(
                "Each point's cell from its index: X varies fastest, then Y, then Z, as Newton's add_particle_grid orders them."
            )
        with g.Frame("Centre"):
            _string_1 = g.String(
                "With Centre on, the grid shifts back by half its size in X and Y so Origin is its middle; Z still starts at Origin, so a centred grid stacks upwards from it."
            )
            switch = centre.switch.vector(
                (0.0, 0.0, 0.0),
                g.CombineXYZ(x=count_x - 1, y=count_y - 1).o.vector * (spacing / -2.0),
            )
        vector_math = (
            origin
            + switch
            + combine_xyz.o.vector * spacing
            + g.RandomValue.vector(jitter * -1.0, jitter, seed=seed)
        )
        points_1 = g.Points(count=count_x * count_y * count_z, position=vector_math)

        points_1 >> points


ASSET = PointGrid

ASSET_METADATA = {
    "description": "A jittered 3D grid of points, as Newton's add_particle_grid",
    "catalog_id": "1d6063ad-2abb-4cca-9bb6-26d4a371d155",
}
