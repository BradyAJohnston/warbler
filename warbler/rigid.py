import databpy as db
import newton
import warp as wp
from bpy.types import Object

from . import utils
from .props import WarblerObjectProperties, object_properties
from .utils import quat_to_blender


class RigidObject(db.BlenderObjectBase):
    def __init__(self, obj: Object):
        super().__init__(obj)
        self.object.rotation_mode = "QUATERNION"

    @property
    def props(self) -> WarblerObjectProperties:
        return object_properties(self.object)

    @property
    def is_active(self) -> bool:
        return self.props.is_active

    def wp_transform(self) -> wp.transform:
        return utils.wp_transform(self.object)

    def shape_config(self) -> newton.ModelBuilder.ShapeConfig:
        prop_names: list[str] = [x for x in dir(self.props) if x.startswith("rigid_")]
        prop_dict = {
            name.replace("rigid_", ""): getattr(self.props, name) for name in prop_names
        }

        return newton.ModelBuilder.ShapeConfig(**prop_dict)

    def transform_from_wp(self, transform: wp.transform) -> None:
        self.object.location = transform[0:3]
        self.object.rotation_quaternion = quat_to_blender(transform[3:7])
