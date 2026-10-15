# Node-group asset "Set Binding Sites" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
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


class SetBindingSites(AssetGeometryGroup):
    """
    Mark vertices of a mesh to be instanced as rigid bodies as binding sites, which attract particles with an affinity

    Parameters
    ----------
    mesh : InputGeometry
        The mesh each instance (rigid body) is made from
    selection : InputBoolean
        Vertices that bind: all of them for non-specific sticking, or a patch for a binding domain
    type : InputInteger
        Interaction type of these sites; an Interaction row says which types attract each other
    strength : InputFloat
        Multiplies the depth of every interaction at these sites

    Inputs
    ------
    i.mesh : GeometrySocket
        The mesh each instance (rigid body) is made from
    i.selection : BooleanSocket
        Vertices that bind: all of them for non-specific sticking, or a patch for a binding domain
    i.type : IntegerSocket
        Interaction type of these sites; an Interaction row says which types attract each other
    i.strength : FloatSocket
        Multiplies the depth of every interaction at these sites

    Outputs
    -------
    o.mesh : GeometrySocket
        The mesh each instance (rigid body) is made from
    """

    _name = "Set Binding Sites"
    _asset_name = "Set Binding Sites"
    _library = PackageLibrary(__file__, "../../assets/nodes.blend")
    _color_tag = "GEOMETRY"
    _tree_properties = {
        "description": "Mark vertices of a mesh to be instanced as rigid bodies as binding sites, which attract particles with an affinity"
    }

    class _Inputs(SocketAccessor):
        mesh: GeometrySocket
        """The mesh each instance (rigid body) is made from"""
        selection: BooleanSocket
        """Vertices that bind: all of them for non-specific sticking, or a patch for a binding domain"""
        type: IntegerSocket
        """Interaction type of these sites; an Interaction row says which types attract each other"""
        strength: FloatSocket
        """Multiplies the depth of every interaction at these sites"""

    class _Outputs(SocketAccessor):
        mesh: GeometrySocket
        """The mesh each instance (rigid body) is made from"""

    if TYPE_CHECKING:

        @property
        def i(self) -> _Inputs: ...
        @property
        def o(self) -> _Outputs: ...

    def __init__(
        self,
        mesh: InputGeometry = None,
        selection: InputBoolean = True,
        type: InputInteger = 1,
        strength: InputFloat = 1.0,
    ):
        super().__init__(Mesh=mesh, Selection=selection, Type=type, Strength=strength)

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        mesh = tree.inputs.geometry(
            "Mesh", description="The mesh each instance (rigid body) is made from"
        )
        selection = tree.inputs.boolean(
            "Selection",
            True,
            description="Vertices that bind: all of them for non-specific sticking, or a patch for a binding domain",
            hide_value=True,
        )
        type = tree.inputs.integer(
            "Type",
            1,
            description="Interaction type of these sites; an Interaction row says which types attract each other",
            min_value=1,
        )
        strength = tree.inputs.float(
            "Strength",
            1.0,
            description="Multiplies the depth of every interaction at these sites",
            min_value=0.0,
        )
        mesh_1 = tree.outputs.geometry(
            "Mesh", description="The mesh each instance (rigid body) is made from"
        )

        with g.Frame("Binding sites"):
            (
                mesh
                >> g.StoreNamedAttribute.point.integer(
                    selection=selection, name="binding_type", value=type
                )
                >> g.StoreNamedAttribute.point.float(
                    selection=selection, name="binding", value=strength
                )
                >> mesh_1
            )
            _string = g.String(
                "Warbler reads the POINT binding_type and binding strength of the instanced mesh's vertices; each vertex with both above 0 is a binding site, carried with the rigid body. Unselected vertices keep the values they had (0 if none)."
            )


ASSET = SetBindingSites

ASSET_METADATA = {
    "description": "Mark vertices of a mesh to be instanced as rigid bodies as binding sites, which attract particles with an affinity",
    "catalog_id": "7b42a9d6-706e-463d-aea2-e0c91fbe1f8d",
}
