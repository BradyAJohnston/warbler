# Node-group asset "Set Pinned" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
# _build_group() is the source of truth: the docstring, __init__ and accessors are regenerated from it on the next dump.
from typing import TYPE_CHECKING

from bpy.types import GeometryNodeTree
from nodebpy import TreeBuilder
from nodebpy import geometry as g
from nodebpy.builder import (
    AssetGeometryGroup,
    BooleanSocket,
    GeometrySocket,
    PackageLibrary,
    SocketAccessor,
)
from nodebpy.types import InputBoolean, InputGeometry


class SetPinned(AssetGeometryGroup):
    """
    Pin points: they follow the source geometry every frame instead of being simulated, so animated pins drag the rest along

    Parameters
    ----------
    geometry : InputGeometry
        Geometry
    pinned : InputBoolean
        Points held in place, following the source geometry

    Inputs
    ------
    i.geometry : GeometrySocket
        Geometry
    i.pinned : BooleanSocket
        Points held in place, following the source geometry

    Outputs
    -------
    o.geometry : GeometrySocket
        Geometry
    """

    _name = "Set Pinned"
    _asset_name = "Set Pinned"
    _library = PackageLibrary(__file__, "../../assets/nodes.blend")
    _color_tag = "GEOMETRY"
    _tree_properties = {
        "description": "Pin points: they follow the source geometry every frame instead of being simulated, so animated pins drag the rest along"
    }

    class _Inputs(SocketAccessor):
        geometry: GeometrySocket
        """Geometry"""
        pinned: BooleanSocket
        """Points held in place, following the source geometry"""

    class _Outputs(SocketAccessor):
        geometry: GeometrySocket
        """Geometry"""

    if TYPE_CHECKING:

        @property
        def i(self) -> _Inputs: ...
        @property
        def o(self) -> _Outputs: ...

    def __init__(
        self,
        geometry: InputGeometry = None,
        pinned: InputBoolean = True,
    ):
        super().__init__(Geometry=geometry, Pinned=pinned)

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        geometry = tree.inputs.geometry("Geometry")
        pinned = tree.inputs.boolean(
            "Pinned",
            True,
            description="Points held in place, following the source geometry",
            hide_value=True,
        )
        geometry_1 = tree.outputs.geometry("Geometry")

        (
            geometry
            >> g.StoreNamedAttribute.point.boolean(name="pinned", value=pinned)
            >> geometry_1
        )


ASSET = SetPinned

ASSET_METADATA = {
    "description": "Pin points: they follow the source geometry every frame instead of being simulated, so animated pins drag the rest along",
    "catalog_id": "ba19136a-743e-4b12-acd6-6f4da209c863",
}
