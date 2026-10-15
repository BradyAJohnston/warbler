# Node-group asset "Set Attached" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
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


class SetAttached(AssetGeometryGroup):
    """
    Attach points to the nearest rigid body, where they start, with a spring (VBD only)

    Parameters
    ----------
    geometry : InputGeometry
        Geometry
    attached : InputBoolean
        Points held to the nearest rigid body

    Inputs
    ------
    i.geometry : GeometrySocket
        Geometry
    i.attached : BooleanSocket
        Points held to the nearest rigid body

    Outputs
    -------
    o.geometry : GeometrySocket
        Geometry
    """

    _name = "Set Attached"
    _asset_name = "Set Attached"
    _library = PackageLibrary(__file__, "../../assets/nodes.blend")
    _color_tag = "GEOMETRY"
    _tree_properties = {
        "description": "Attach points to the nearest rigid body, where they start, with a spring (VBD only)"
    }

    class _Inputs(SocketAccessor):
        geometry: GeometrySocket
        """Geometry"""
        attached: BooleanSocket
        """Points held to the nearest rigid body"""

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
        attached: InputBoolean = True,
    ):
        super().__init__(Geometry=geometry, Attached=attached)

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        geometry = tree.inputs.geometry("Geometry")
        attached = tree.inputs.boolean(
            "Attached",
            True,
            description="Points held to the nearest rigid body",
            hide_value=True,
        )
        geometry_1 = tree.outputs.geometry("Geometry")

        (
            geometry
            >> g.StoreNamedAttribute.point.boolean(name="attach", value=attached)
            >> geometry_1
        )


ASSET = SetAttached

ASSET_METADATA = {
    "description": "Attach points to the nearest rigid body, where they start, with a spring (VBD only)",
    "catalog_id": "ba19136a-743e-4b12-acd6-6f4da209c863",
}
