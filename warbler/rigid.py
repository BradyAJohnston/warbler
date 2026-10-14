import databpy as db
import newton
import numpy as np
import warp as wp
from mathutils import Matrix, Quaternion

from . import utils
from .props import WarblerObjectProperties, object_properties
from .utils import quat_to_blender


class RigidObject(db.BlenderObjectBase):
    @property
    def props(self) -> WarblerObjectProperties:
        return object_properties(self.object)

    @property
    def is_active(self) -> bool:
        return self.props.is_active

    def half_extents(self) -> np.ndarray:
        """Half the world-space size of the object's bounding box.

        Local bounds scaled by the world matrix, so parent scale counts too.
        """
        bounds = np.array(self.object.bound_box, dtype=np.float64)
        size = bounds.max(axis=0) - bounds.min(axis=0)
        return size * np.abs(np.array(self.object.matrix_world.to_scale())) / 2.0

    def wp_transform(self) -> wp.transform:
        return utils.wp_transform(self.object)

    def shape_config(self) -> newton.ModelBuilder.ShapeConfig:
        prop_names: list[str] = [x for x in dir(self.props) if x.startswith("rigid_")]
        prop_dict = {
            name.replace("rigid_", ""): getattr(self.props, name) for name in prop_names
        }

        return newton.ModelBuilder.ShapeConfig(**prop_dict)

    def transform_from_wp(self, transform: wp.transform) -> None:
        """Place the object at a world-space sim transform, keeping its scale.

        Writing ``matrix_world`` lets Blender solve the local transform, so
        parented objects and any rotation mode work.
        """
        scale = self.object.matrix_world.to_scale()
        self.object.matrix_world = Matrix.LocRotScale(
            [float(x) for x in transform[0:3]],
            Quaternion(quat_to_blender(transform[3:7])),
            [scale.x, scale.y, scale.z],
        )
