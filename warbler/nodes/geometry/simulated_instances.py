# Node-group asset "Simulated Instances" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
# _build_group() is the source of truth: the docstring, __init__ and accessors are regenerated from it on the next dump.
from typing import TYPE_CHECKING

from bpy.types import GeometryNodeTree
from nodebpy import TreeBuilder
from nodebpy import geometry as g
from nodebpy.builder import (
    AssetGeometryGroup,
    GeometrySocket,
    ObjectSocket,
    PackageLibrary,
    SocketAccessor,
)
from nodebpy.types import InputGeometry, InputObject


class SimulatedInstances(AssetGeometryGroup):
    """
    Re-place the source's instances at their simulated transforms, read from the instances output's points

    Parameters
    ----------
    points : InputGeometry
        The instances output's points, with instance_transform
    source : InputObject
        The simulation's source

    Inputs
    ------
    i.points : GeometrySocket
        The instances output's points, with instance_transform
    i.source : ObjectSocket
        The simulation's source

    Outputs
    -------
    o.instances : GeometrySocket
        Instances
    """

    _name = "Simulated Instances"
    _asset_name = "Simulated Instances"
    _library = PackageLibrary(__file__, "../../assets/nodes.blend")
    _color_tag = "GEOMETRY"
    _tree_properties = {
        "description": "Re-place the source's instances at their simulated transforms, read from the instances output's points"
    }

    class _Inputs(SocketAccessor):
        points: GeometrySocket
        """The instances output's points, with instance_transform"""
        source: ObjectSocket
        """The simulation's source"""

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
        points: InputGeometry = None,
        source: InputObject = None,
    ):
        super().__init__(Points=points, Source=source)

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        points = tree.inputs.geometry(
            "Points",
            description="The instances output's points, with instance_transform",
        )
        source = tree.inputs.object("Source", description="The simulation's source")
        instances = tree.outputs.geometry("Instances")

        with g.Frame("Simulated transform of each instance"):
            sample_index = g.SampleIndex.point.matrix(
                points,
                g.NamedAttribute.matrix("instance_transform").o.attribute,
                g.Index(),
            )
            set_instance_transform = g.SetInstanceTransform(
                instances=g.SeparateComponents(
                    g.ObjectInfo(source).o.geometry
                ).o.instances,
                transform=sample_index,
            )
            _string = g.String(
                "Warbler writes one point per source instance, in order, carrying its world transform after the latest frame, so instance i takes point i's instance_transform. Object Info keeps the source's own (original) space and the output object sits at the world origin, so the world transform places each instance directly."
            )

        set_instance_transform >> instances


ASSET = SimulatedInstances

ASSET_METADATA = {
    "description": "Re-place the source's instances at their simulated transforms, read from the instances output's points",
    "catalog_id": "54890066-5295-4dce-b842-0370f7024048",
}
