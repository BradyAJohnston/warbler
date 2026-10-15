# Node-group asset "Set On Surface" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
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
    PackageLibrary,
    SocketAccessor,
)
from nodebpy.types import InputBoolean, InputFloat, InputGeometry


class SetOnSurface(AssetGeometryGroup):
    """
    Hold points on the simulation's Surface mesh, at an offset along its normal

    Parameters
    ----------
    geometry : InputGeometry
        Geometry
    selection : InputBoolean
        Points held on the surface
    offset : InputFloat
        Distance along the surface normal

    Inputs
    ------
    i.geometry : GeometrySocket
        Geometry
    i.selection : BooleanSocket
        Points held on the surface
    i.offset : FloatSocket
        Distance along the surface normal

    Outputs
    -------
    o.geometry : GeometrySocket
        Geometry
    """

    _name = "Set On Surface"
    _asset_name = "Set On Surface"
    _library = PackageLibrary(__file__, "../../assets/nodes.blend")
    _color_tag = "GEOMETRY"
    _tree_properties = {
        "description": "Hold points on the simulation's Surface mesh, at an offset along its normal"
    }

    class _Inputs(SocketAccessor):
        geometry: GeometrySocket
        """Geometry"""
        selection: BooleanSocket
        """Points held on the surface"""
        offset: FloatSocket
        """Distance along the surface normal"""

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
        selection: InputBoolean = True,
        offset: InputFloat = 0.0,
    ):
        super().__init__(Geometry=geometry, Selection=selection, Offset=offset)

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        geometry = tree.inputs.geometry("Geometry")
        selection = tree.inputs.boolean(
            "Selection", True, description="Points held on the surface", hide_value=True
        )
        offset = tree.inputs.float(
            "Offset", 0.0, description="Distance along the surface normal"
        )
        geometry_1 = tree.outputs.geometry("Geometry")

        (
            geometry
            >> g.StoreNamedAttribute.point.boolean(name="on_surface", value=selection)
            >> g.StoreNamedAttribute.point.float(name="surface_offset", value=offset)
            >> geometry_1
        )


ASSET = SetOnSurface

ASSET_METADATA = {
    "description": "Hold points on the simulation's Surface mesh, at an offset along its normal",
    "catalog_id": "ba19136a-743e-4b12-acd6-6f4da209c863",
}
