# Node-group asset "Cloth Grid" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
# _build_group() is the source of truth: the docstring, __init__ and accessors are regenerated from it on the next dump.
import math
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
)
from nodebpy.types import InputFloat, InputInteger


class ClothGrid(AssetGeometryGroup):
    """
    A cloth sheet laid out as Newton's add_cloth_grid: corner at the origin, first edge along Y at x = 0, raised to Height

    Parameters
    ----------
    count_x : InputInteger
        Cells along X
    count_y : InputInteger
        Cells along Y
    cell_size : InputFloat
        Cell Size
    height : InputFloat
        Height

    Inputs
    ------
    i.count_x : IntegerSocket
        Cells along X
    i.count_y : IntegerSocket
        Cells along Y
    i.cell_size : FloatSocket
        Cell Size
    i.height : FloatSocket
        Height

    Outputs
    -------
    o.mesh : GeometrySocket
        Mesh
    o.edge : BooleanSocket
        Vertices of the edge along Y at x = 0, to pin or attach
    """

    _name = "Cloth Grid"
    _asset_name = "Cloth Grid"
    _library = PackageLibrary(__file__, "../../assets/nodes.blend")
    _color_tag = "GEOMETRY"
    _tree_properties = {
        "description": "A cloth sheet laid out as Newton's add_cloth_grid: corner at the origin, first edge along Y at x = 0, raised to Height"
    }

    class _Inputs(SocketAccessor):
        count_x: IntegerSocket
        """Cells along X"""
        count_y: IntegerSocket
        """Cells along Y"""
        cell_size: FloatSocket
        """Cell Size"""
        height: FloatSocket
        """Height"""

    class _Outputs(SocketAccessor):
        mesh: GeometrySocket
        """Mesh"""
        edge: BooleanSocket
        """Vertices of the edge along Y at x = 0, to pin or attach"""

    if TYPE_CHECKING:

        @property
        def i(self) -> _Inputs: ...
        @property
        def o(self) -> _Outputs: ...

    def __init__(
        self,
        count_x: InputInteger = 64,
        count_y: InputInteger = 32,
        cell_size: InputFloat = 0.1,
        height: InputFloat = 4.0,
    ):
        super().__init__(
            **{
                "Count X": count_x,
                "Count Y": count_y,
                "Cell Size": cell_size,
                "Height": height,
            }
        )

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        count_x = tree.inputs.integer(
            "Count X", 64, description="Cells along X", min_value=1
        )
        count_y = tree.inputs.integer(
            "Count Y", 32, description="Cells along Y", min_value=1
        )
        cell_size = tree.inputs.float("Cell Size", 0.1, min_value=0.001)
        height = tree.inputs.float("Height", 4.0)
        mesh = tree.outputs.geometry("Mesh")
        edge = tree.outputs.boolean(
            "Edge",
            description="Vertices of the edge along Y at x = 0, to pin or attach",
        )

        math_1 = count_x * cell_size
        math_2 = count_y * cell_size
        with g.Frame("Newton's cloth grid layout"):
            _string = g.String(
                "Grid is centred on the origin; Newton's add_cloth_grid puts its corner there and turns it 90 degrees about Z. Moving the corner to the origin first and then rotating leaves the first edge along Y at x = 0."
            )
            (
                g.Grid(
                    size_x=math_1,
                    size_y=math_2,
                    vertices_x=count_x + 1,
                    vertices_y=count_y + 1,
                )
                >> g.TransformGeometry(
                    translation=g.CombineXYZ(x=math_1 / 2.0, y=math_2 / 2.0)
                )
                >> g.TransformGeometry(
                    translation=g.CombineXYZ(z=height), rotation=(0.0, 0.0, math.pi / 2)
                )
                >> mesh
            )
        with g.Frame("Edge"):
            (g.Position().o.position.y < cell_size * 0.01) >> edge
            _string_1 = g.String(
                "A field, evaluated wherever Edge is used: the vertices on the first edge, within a hundredth of a cell of y = 0."
            )


ASSET = ClothGrid

ASSET_METADATA = {
    "description": "A cloth sheet laid out as Newton's add_cloth_grid: corner at the origin, first edge along Y at x = 0, raised to Height",
    "catalog_id": "1d6063ad-2abb-4cca-9bb6-26d4a371d155",
}
