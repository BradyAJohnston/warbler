# Node-group asset "Set Velocity" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
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
    VectorSocket,
)
from nodebpy.types import InputBoolean, InputGeometry, InputVector


class SetVelocity(AssetGeometryGroup):
    """
    Starting velocity of particles (points) and rigid bodies (instances)

    Parameters
    ----------
    geometry : InputGeometry
        Geometry
    selection : InputBoolean
        Selection
    velocity : InputVector
        Velocity at the start of the simulation

    Inputs
    ------
    i.geometry : GeometrySocket
        Geometry
    i.selection : BooleanSocket
        Selection
    i.velocity : VectorSocket
        Velocity at the start of the simulation

    Outputs
    -------
    o.geometry : GeometrySocket
        Geometry
    """

    _name = "Set Velocity"
    _asset_name = "Set Velocity"
    _library = PackageLibrary(__file__, "../../assets/nodes.blend")
    _color_tag = "GEOMETRY"
    _tree_properties = {
        "description": "Starting velocity of particles (points) and rigid bodies (instances)"
    }

    class _Inputs(SocketAccessor):
        geometry: GeometrySocket
        """Geometry"""
        selection: BooleanSocket
        """Selection"""
        velocity: VectorSocket
        """Velocity at the start of the simulation"""

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
        velocity: InputVector = None,
    ):
        super().__init__(Geometry=geometry, Selection=selection, Velocity=velocity)

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        geometry = tree.inputs.geometry("Geometry")
        selection = tree.inputs.boolean("Selection", True, hide_value=True)
        velocity = tree.inputs.vector(
            "Velocity",
            (0.0, 0.0, 0.0),
            description="Velocity at the start of the simulation",
            subtype="VELOCITY",
        )
        geometry_1 = tree.outputs.geometry("Geometry")

        with g.Frame("Both domains"):
            (
                geometry
                >> g.StoreNamedAttribute.point.vector(
                    selection=selection, name="velocity", value=velocity
                )
                >> g.StoreNamedAttribute.instance.vector(
                    selection=selection, name="velocity", value=velocity
                )
                >> geometry_1
            )
            _string = g.String(
                "Points become particles and instances become rigid bodies, each reading velocity from its own domain, so it's stored on both."
            )


ASSET = SetVelocity

ASSET_METADATA = {
    "description": "Starting velocity of particles (points) and rigid bodies (instances)",
    "catalog_id": "ba19136a-743e-4b12-acd6-6f4da209c863",
}
