# Node-group asset "Set Density" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
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


class SetDensity(AssetGeometryGroup):
    """
    Density each instance's rigid body mass and inertia are computed from

    Parameters
    ----------
    instances : InputGeometry
        Instances
    selection : InputBoolean
        Selection
    density : InputFloat
        Mass per unit volume (water is 1000)

    Inputs
    ------
    i.instances : GeometrySocket
        Instances
    i.selection : BooleanSocket
        Selection
    i.density : FloatSocket
        Mass per unit volume (water is 1000)

    Outputs
    -------
    o.instances : GeometrySocket
        Instances
    """

    _name = "Set Density"
    _asset_name = "Set Density"
    _library = PackageLibrary(__file__, "../../assets/nodes.blend")
    _color_tag = "GEOMETRY"
    _tree_properties = {
        "description": "Density each instance's rigid body mass and inertia are computed from"
    }

    class _Inputs(SocketAccessor):
        instances: GeometrySocket
        """Instances"""
        selection: BooleanSocket
        """Selection"""
        density: FloatSocket
        """Mass per unit volume (water is 1000)"""

    class _Outputs(SocketAccessor):
        instances: GeometrySocket
        """Instances"""

    if TYPE_CHECKING:

        @property
        def i(self) -> _Inputs: ...
        @property
        def o(self) -> _Outputs: ...

    def __init__(
        self,
        instances: InputGeometry = None,
        selection: InputBoolean = True,
        density: InputFloat = 1000.0,
    ):
        super().__init__(Instances=instances, Selection=selection, Density=density)

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        instances = tree.inputs.geometry("Instances")
        selection = tree.inputs.boolean("Selection", True, hide_value=True)
        density = tree.inputs.float(
            "Density",
            1000.0,
            description="Mass per unit volume (water is 1000)",
            min_value=0.001,
        )
        instances_1 = tree.outputs.geometry("Instances")

        (
            instances
            >> g.StoreNamedAttribute.instance.float(
                selection=selection, name="density", value=density
            )
            >> instances_1
        )


ASSET = SetDensity

ASSET_METADATA = {
    "description": "Density each instance's rigid body mass and inertia are computed from",
    "catalog_id": "e169a26f-7114-4f10-a7cb-559a01c4a7be",
}
