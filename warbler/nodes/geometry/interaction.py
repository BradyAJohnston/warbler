# Node-group asset "Interaction" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
# _build_group() is the source of truth: the docstring, __init__ and accessors are regenerated from it on the next dump.
from typing import TYPE_CHECKING

from bpy.types import GeometryNodeTree
from nodebpy import TreeBuilder
from nodebpy import geometry as g
from nodebpy.builder import (
    AssetGeometryGroup,
    FloatSocket,
    GeometrySocket,
    IntegerSocket,
    PackageLibrary,
    SocketAccessor,
)
from nodebpy.types import InputFloat, InputInteger


class Interaction(AssetGeometryGroup):
    """
    One row of the interaction table: how strongly two interaction types attract. Join rows into the simulation's source

    Parameters
    ----------
    type_a : InputInteger
        One interaction type
    type_b : InputInteger
        The type it attracts (may be the same)
    depth : InputFloat
        Depth of the attraction, in the units of Temperature: well above it sticks, near it binds and unbinds
    range : InputFloat
        Width of the attraction; it reaches 3 times this

    Inputs
    ------
    i.type_a : IntegerSocket
        One interaction type
    i.type_b : IntegerSocket
        The type it attracts (may be the same)
    i.depth : FloatSocket
        Depth of the attraction, in the units of Temperature: well above it sticks, near it binds and unbinds
    i.range : FloatSocket
        Width of the attraction; it reaches 3 times this

    Outputs
    -------
    o.interaction : GeometrySocket
        Join into the simulation's source geometry
    """

    _name = "Interaction"
    _asset_name = "Interaction"
    _library = PackageLibrary(__file__, "../../assets/nodes.blend")
    _color_tag = "INPUT"
    _tree_properties = {
        "description": "One row of the interaction table: how strongly two interaction types attract. Join rows into the simulation's source"
    }

    class _Inputs(SocketAccessor):
        type_a: IntegerSocket
        """One interaction type"""
        type_b: IntegerSocket
        """The type it attracts (may be the same)"""
        depth: FloatSocket
        """Depth of the attraction, in the units of Temperature: well above it sticks, near it binds and unbinds"""
        range: FloatSocket
        """Width of the attraction; it reaches 3 times this"""

    class _Outputs(SocketAccessor):
        interaction: GeometrySocket
        """Join into the simulation's source geometry"""

    if TYPE_CHECKING:

        @property
        def i(self) -> _Inputs: ...
        @property
        def o(self) -> _Outputs: ...

    def __init__(
        self,
        type_a: InputInteger = 1,
        type_b: InputInteger = 2,
        depth: InputFloat = 0.5,
        range: InputFloat = 0.05,
    ):
        super().__init__(
            **{"Type A": type_a, "Type B": type_b, "Depth": depth, "Range": range}
        )

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        type_a = tree.inputs.integer(
            "Type A", 1, description="One interaction type", min_value=1
        )
        type_b = tree.inputs.integer(
            "Type B",
            2,
            description="The type it attracts (may be the same)",
            min_value=1,
        )
        depth = tree.inputs.float(
            "Depth",
            0.5,
            description="Depth of the attraction, in the units of Temperature: well above it sticks, near it binds and unbinds",
            min_value=0.0,
        )
        range = tree.inputs.float(
            "Range",
            0.05,
            description="Width of the attraction; it reaches 3 times this",
            min_value=0.0001,
            subtype="DISTANCE",
        )
        interaction = tree.outputs.geometry(
            "Interaction", description="Join into the simulation's source geometry"
        )

        with g.Frame("Table row"):
            (
                g.Points()
                >> g.StoreNamedAttribute.point.boolean(name="interaction", value=True)
                >> g.StoreNamedAttribute.point.integer(
                    name="interaction_a", value=type_a
                )
                >> g.StoreNamedAttribute.point.integer(
                    name="interaction_b", value=type_b
                )
                >> g.StoreNamedAttribute.point.float(
                    name="interaction_depth", value=depth
                )
                >> g.StoreNamedAttribute.point.float(
                    name="interaction_range", value=range
                )
                >> interaction
            )
            _string = g.String(
                "Warbler takes points flagged interaction out of the particles and reads them as the interaction table. Types attract only if a row lists them; a row works both ways. Rows are re-read every frame, so Depth and Range can be animated."
            )


ASSET = Interaction

ASSET_METADATA = {
    "description": "One row of the interaction table: how strongly two interaction types attract. Join rows into the simulation's source",
    "catalog_id": "7b42a9d6-706e-463d-aea2-e0c91fbe1f8d",
}
