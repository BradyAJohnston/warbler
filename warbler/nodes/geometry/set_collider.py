# Node-group asset "Set Collider" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
# _build_group() is the source of truth: the docstring, __init__ and accessors are regenerated from it on the next dump.
from typing import TYPE_CHECKING, Literal

from bpy.types import GeometryNodeTree
from nodebpy import TreeBuilder
from nodebpy import geometry as g
from nodebpy.builder import (
    AssetGeometryGroup,
    BooleanSocket,
    GeometrySocket,
    MenuSocket,
    PackageLibrary,
    SocketAccessor,
)
from nodebpy.types import InputBoolean, InputGeometry, InputMenu


class SetCollider(AssetGeometryGroup):
    """
    Collision shape of each instance's rigid body. Primitives fit the instanced geometry's bounding box; Convex Hull and Mesh use its vertices and faces

    Parameters
    ----------
    instances : InputGeometry
        Instances, each simulated as one rigid body
    selection : InputBoolean
        Selection
    shape : InputMenu | Literal["Convex Hull", "Box", "Sphere", "Capsule", "Mesh"]
        Collision shape of the rigid body

    Inputs
    ------
    i.instances : GeometrySocket
        Instances, each simulated as one rigid body
    i.selection : BooleanSocket
        Selection
    i.shape : MenuSocket
        Collision shape of the rigid body

    Outputs
    -------
    o.instances : GeometrySocket
        Instances, each simulated as one rigid body
    """

    _name = "Set Collider"
    _asset_name = "Set Collider"
    _library = PackageLibrary(__file__, "../../assets/nodes.blend")
    _color_tag = "GEOMETRY"
    _tree_properties = {
        "description": "Collision shape of each instance's rigid body. Primitives fit the instanced geometry's bounding box; Convex Hull and Mesh use its vertices and faces"
    }

    class _Inputs(SocketAccessor):
        instances: GeometrySocket
        """Instances, each simulated as one rigid body"""
        selection: BooleanSocket
        """Selection"""
        shape: MenuSocket
        """Collision shape of the rigid body"""

    class _Outputs(SocketAccessor):
        instances: GeometrySocket
        """Instances, each simulated as one rigid body"""

    if TYPE_CHECKING:

        @property
        def i(self) -> _Inputs: ...
        @property
        def o(self) -> _Outputs: ...

    def __init__(
        self,
        instances: InputGeometry = None,
        selection: InputBoolean = True,
        shape: InputMenu
        | Literal["Convex Hull", "Box", "Sphere", "Capsule", "Mesh"] = "Convex Hull",
    ):
        super().__init__(Instances=instances, Selection=selection, Shape=shape)

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        instances = tree.inputs.geometry(
            "Instances", description="Instances, each simulated as one rigid body"
        )
        selection = tree.inputs.boolean("Selection", True, hide_value=True)
        shape = tree.inputs.menu(
            "Shape", "Convex Hull", description="Collision shape of the rigid body"
        )
        instances_1 = tree.outputs.geometry(
            "Instances", description="Instances, each simulated as one rigid body"
        )

        with g.Frame("Collider values"):
            menu_switch = g.MenuSwitch.integer(
                shape,
                {"Convex Hull": 0, "Box": 1, "Sphere": 2, "Capsule": 3, "Mesh": 4},
            )
            _string = g.String(
                "The integers are the INSTANCE collider values Warbler reads: 0 Convex Hull, 1 Box, 2 Sphere, 3 Capsule, 4 Mesh. Keep them in the order of COLLIDERS in warbler/simulation.py."
            )
        (
            instances
            >> g.StoreNamedAttribute.instance.integer(
                selection=selection, name="collider", value=menu_switch.o.output
            )
            >> instances_1
        )


ASSET = SetCollider

ASSET_METADATA = {
    "description": "Collision shape of each instance's rigid body. Primitives fit the instanced geometry's bounding box; Convex Hull and Mesh use its vertices and faces",
    "catalog_id": "e169a26f-7114-4f10-a7cb-559a01c4a7be",
}
