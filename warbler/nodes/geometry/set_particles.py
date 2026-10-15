# Node-group asset "Set Particles" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
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


class SetParticles(AssetGeometryGroup):
    """
    Store the mass and collision radius Warbler reads for each particle

    Parameters
    ----------
    geometry : InputGeometry
        Points, curves or a mesh to simulate
    selection : InputBoolean
        Points to set
    mass : InputFloat
        Particle mass. 0 makes the particle immovable
    radius : InputFloat
        Collision radius

    Inputs
    ------
    i.geometry : GeometrySocket
        Points, curves or a mesh to simulate
    i.selection : BooleanSocket
        Points to set
    i.mass : FloatSocket
        Particle mass. 0 makes the particle immovable
    i.radius : FloatSocket
        Collision radius

    Outputs
    -------
    o.geometry : GeometrySocket
        Points, curves or a mesh to simulate
    """

    _name = "Set Particles"
    _asset_name = "Set Particles"
    _library = PackageLibrary(__file__, "../../assets/nodes.blend")
    _color_tag = "GEOMETRY"
    _tree_properties = {
        "description": "Store the mass and collision radius Warbler reads for each particle"
    }

    class _Inputs(SocketAccessor):
        geometry: GeometrySocket
        """Points, curves or a mesh to simulate"""
        selection: BooleanSocket
        """Points to set"""
        mass: FloatSocket
        """Particle mass. 0 makes the particle immovable"""
        radius: FloatSocket
        """Collision radius"""

    class _Outputs(SocketAccessor):
        geometry: GeometrySocket
        """Points, curves or a mesh to simulate"""

    if TYPE_CHECKING:

        @property
        def i(self) -> _Inputs: ...
        @property
        def o(self) -> _Outputs: ...

    def __init__(
        self,
        geometry: InputGeometry = None,
        selection: InputBoolean = True,
        mass: InputFloat = 1.0,
        radius: InputFloat = 0.05,
    ):
        super().__init__(
            Geometry=geometry, Selection=selection, Mass=mass, Radius=radius
        )

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        geometry = tree.inputs.geometry(
            "Geometry", description="Points, curves or a mesh to simulate"
        )
        selection = tree.inputs.boolean(
            "Selection", True, description="Points to set", hide_value=True
        )
        mass = tree.inputs.float(
            "Mass",
            1.0,
            description="Particle mass. 0 makes the particle immovable",
            min_value=0.0,
        )
        radius = tree.inputs.float(
            "Radius", 0.05, description="Collision radius", min_value=0.0
        )
        geometry_1 = tree.outputs.geometry(
            "Geometry", description="Points, curves or a mesh to simulate"
        )

        (
            geometry
            >> g.StoreNamedAttribute.point.float(
                selection=selection, name="mass", value=mass
            )
            >> g.StoreNamedAttribute.point.float(
                selection=selection, name="radius", value=radius
            )
            >> geometry_1
        )


ASSET = SetParticles

ASSET_METADATA = {
    "description": "Store the mass and collision radius Warbler reads for each particle",
    "catalog_id": "ba19136a-743e-4b12-acd6-6f4da209c863",
}
