# Node-group asset "Set Affinity" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
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
)
from nodebpy.types import InputBoolean, InputFloat, InputGeometry, InputInteger


class SetAffinity(AssetGeometryGroup):
    """
    Interaction type of particles, and how strongly they are attracted to binding sites on rigid bodies

    Parameters
    ----------
    geometry : InputGeometry
        Geometry
    selection : InputBoolean
        Selection
    type : InputInteger
        Interaction type of these particles; an Interaction row says which types attract each other
    affinity : InputFloat
        Multiplies the depth of every interaction of these particles. 0 never binds

    Inputs
    ------
    i.geometry : GeometrySocket
        Geometry
    i.selection : BooleanSocket
        Selection
    i.type : IntegerSocket
        Interaction type of these particles; an Interaction row says which types attract each other
    i.affinity : FloatSocket
        Multiplies the depth of every interaction of these particles. 0 never binds

    Outputs
    -------
    o.geometry : GeometrySocket
        Geometry
    """

    _name = "Set Affinity"
    _asset_name = "Set Affinity"
    _library = PackageLibrary(__file__, "../../assets/nodes.blend")
    _color_tag = "GEOMETRY"
    _tree_properties = {
        "description": "Interaction type of particles, and how strongly they are attracted to binding sites on rigid bodies"
    }

    class _Inputs(SocketAccessor):
        geometry: GeometrySocket
        """Geometry"""
        selection: BooleanSocket
        """Selection"""
        type: IntegerSocket
        """Interaction type of these particles; an Interaction row says which types attract each other"""
        affinity: FloatSocket
        """Multiplies the depth of every interaction of these particles. 0 never binds"""

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
        type: InputInteger = 1,
        affinity: InputFloat = 1.0,
    ):
        super().__init__(
            Geometry=geometry, Selection=selection, Type=type, Affinity=affinity
        )

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        geometry = tree.inputs.geometry("Geometry")
        selection = tree.inputs.boolean("Selection", True, hide_value=True)
        type = tree.inputs.integer(
            "Type",
            1,
            description="Interaction type of these particles; an Interaction row says which types attract each other",
            min_value=1,
        )
        affinity = tree.inputs.float(
            "Affinity",
            1.0,
            description="Multiplies the depth of every interaction of these particles. 0 never binds",
            min_value=0.0,
        )
        geometry_1 = tree.outputs.geometry("Geometry")

        (
            geometry
            >> g.StoreNamedAttribute.point.integer(
                selection=selection, name="affinity_type", value=type
            )
            >> g.StoreNamedAttribute.point.float(
                selection=selection, name="affinity", value=affinity
            )
            >> geometry_1
        )


ASSET = SetAffinity

ASSET_METADATA = {
    "description": "Interaction type of particles, and how strongly they are attracted to binding sites on rigid bodies",
    "catalog_id": "7b42a9d6-706e-463d-aea2-e0c91fbe1f8d",
}
