from bpy.types import Object
from .geometryset import GeometrySet
import bpy
import warp as wp
import newton
import numpy as np
import databpy as db
from .utils import (
    smooth_lerp,
    blender_rotation,
    wp_transform,
)
from . import rigid
from .props import SimulationListItem
from uuid import uuid1
from typing import TYPE_CHECKING
from abc import ABC
import time


if TYPE_CHECKING:
    from .manager import SimulationManager


class SimulatorBase(ABC):
    @property
    def scene(self) -> bpy.types.Scene:
        return bpy.context.scene

    @property
    def fps(self) -> int:
        return self.scene.render.fps

    @property
    def frame_dt(self) -> float:
        return 1 / self.fps

    @property
    def uuid(self) -> str:
        return self._uuid

    @property
    def manager(self) -> "SimulationManager":
        if self._manager is None:
            raise RuntimeError
        return self._manager

    @property
    def substeps(self) -> int:
        return self.props.substeps

    @substeps.setter
    def substeps(self, value: int) -> None:
        self.props.substeps = value

    @property
    def device(self) -> str:
        return self.props.device

    @device.setter
    def device(self, value: str) -> None:
        self.props.device = value

    @property
    def props(self) -> SimulationListItem:
        return self.manager.sim_items[self.uuid]

    @property
    def is_active(self) -> bool:
        return self.manager.sim_items[self.uuid].is_active

    def __init__(self):
        self._uuid: str = str(uuid1())
        self._manager: SimulationManager | None = None

    def _compile(self) -> None:
        raise NotImplementedError

    def compile(self) -> None:
        self._compile()
        self.props.is_compiled = True


class SimulatorXPBD(SimulatorBase):
    """Physics simulation integrating Newton Physics with Blender.

    Follows Newton's architecture pattern:
    ModelBuilder → Model → State → Solver → Updated State
    """

    @property
    def ke(self) -> float:
        return self.props.spring_ke

    @property
    def kd(self) -> float:
        return self.props.spring_kd

    @property
    def kf(self) -> float:
        return self.props.spring_kf

    @property
    def scale(self) -> float:
        return self.props.scale

    @property
    def objects(self) -> list[Object]:
        return [o for o in self.props.sim_rigid_collection.objects]

    # ============================================================================
    # Initialization
    # ============================================================================

    def __init__(
        self,
    ):
        """Initialize simulation with specified parameters."""
        super().__init__()
        self.clock: int = 0
        self.bob: db.BlenderObject | None = None
        self.search_radius = 1.0

    def _compile(self) -> None:
        if self.props.is_compiled:
            del self.model
        self.build()
        self.finalize()

    def build(self):
        # axis_map = {
        #     (0, 0, 1): newton.Axis.Z,
        #     (0, 1, 0): newton.Axis.Y,
        #     (1, 0, 0): newton.Axis.X,
        # }
        # up_axis = axis_map.get(self.props.ground_plane_vector, newton.Axis.Z)

        self.builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        self.builder.default_particle_radius = 1.0

        self._add_rigid_bodies(self.objects)
        if self.props.use_ground_plane:
            self.builder.add_ground_plane()

        if self.props.particle_source is not None:
            geo = GeometrySet(self.props.particle_source)
            self._add_particles(**geo.pointcloud.to_props())

    def finalize(self):
        self.model: newton.Model = self.builder.finalize(device=self.device)

        self.state_0: newton.State = self.model.state()
        self.state_1: newton.State = self.model.state()

        self.solver = newton.solvers.SolverXPBD(
            model=self.model,
            iterations=self.substeps,
        )
        self.control = self.model.control()
        if self.props.is_compiled:
            bpy.data.objects.remove(self.particle_object.object)
        self.create_pointcloud()

    def _add_rigid_bodies(self, objects: list[bpy.types.Object]):
        """Add Blender objects as rigid bodies to the model."""
        for obj in objects:
            rig = rigid.RigidObject(obj)
            body = self.builder.add_body(mass=1e5, xform=rig.wp_transform())

            if obj.wb.sim_shape == "CUBE":  # type: ignore
                obj.wb.sim_body_index = self.builder.add_shape_box(  # type: ignore
                    body=body,
                    xform=wp_transform(),
                    hx=obj.dimensions[0] / 2.0,
                    hy=obj.dimensions[1] / 2.0,
                    hz=obj.dimensions[2] / 2.0,
                    cfg=rig.shape_config(),
                )
            else:
                print(Warning(f"Unsupported shape {obj.wb.sim_shape}"))  # type: ignore

    def _add_particles(
        self,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
        mass: np.ndarray | None = None,
        radius: np.ndarray | None = None,
    ) -> None:
        if velocity is None:
            velocity = np.zeros(position.shape, dtype=float)
        if mass is None:
            mass = np.repeat(1.0, position.shape[0])
        if radius is None:
            radius = np.repeat(0.1, position.shape[0])

        self.search_radius = max(radius) * 2

        self.builder.add_particles(
            pos=position,  # type: ignore
            vel=velocity,  # type: ignore
            mass=mass,  # type: ignore
            radius=radius,  # type: ignore
        )

    def _add_springs(self):
        """Add spring constraints between consecutive particles."""
        pass
        # for i in range(self.num_particles):
        #     self.edges[i, :] = (i - 1, i)
        #     self.builder.add_spring(i - 1, i, 3, 0.0, 0)

    # ============================================================================
    # Blender ↔ Simulation Synchronization
    # ============================================================================

    def _update_blender_from_simulation(self):
        """Copy physics-controlled body transforms from simulation to Blender."""
        if self.state_0.body_q is None:
            return
        rigid_transforms = self.state_0.body_q.numpy()
        for i, obj in enumerate(self.objects):
            rig = rigid.RigidObject(obj)
            if rig.is_active:
                rig.transform_from_wp(rigid_transforms[i])

    def _update_simulation_from_blender(self):
        """Copy manually-controlled body transforms from Blender to simulation."""
        if self.state_0.body_q is None:
            return

        current_sim_transforms = self.state_0.body_q.numpy()
        body_velocities = None
        if hasattr(self.state_0, "body_qd") and self.state_0.body_qd is not None:
            body_velocities = self.state_0.body_qd.numpy()

        new_transforms = []
        for i, obj in enumerate(self.objects):
            rig = rigid.RigidObject(obj)

            if not rig.is_active:
                transform = self._get_manual_body_transform(
                    obj, current_sim_transforms[i], body_velocities, i
                )
            else:
                transform = wp.transform(
                    wp.vec3(*current_sim_transforms[i, 0:3]),
                    wp.quat(*current_sim_transforms[i, 3:7]),
                )

            new_transforms.append(transform)

        self.state_0.body_q.assign(new_transforms)
        if body_velocities is not None:
            self.state_0.body_qd.assign(body_velocities)

    def _get_manual_body_transform(
        self,
        obj: bpy.types.Object,
        current_sim_transform: np.ndarray,
        body_velocities: np.ndarray | None,
        body_index: int,
    ) -> wp.transform:
        """Calculate transform for manually-controlled body with smoothing.

        Manually-controlled bodies follow Blender positions but are smoothed
        to avoid jarring movements. Their velocity is calculated to push particles
        but then zeroed to prevent self-movement.
        """
        rot = blender_rotation(obj.rotation_quaternion)
        loc = np.array(obj.location)

        # Apply smoothing (except first frame)
        if self.clock != 0:
            loc = smooth_lerp(
                current_sim_transform[0:3],
                loc,
                self.props.rigid_decay_frames,
            )

            # Calculate velocity for particle interaction
            if body_velocities is not None:
                velocity = (loc - current_sim_transform[0:3]) / self.frame_dt
                body_velocities[body_index, 0:3] = velocity

        return wp.transform(wp.vec3(*loc), wp.quat(*rot))

    # ============================================================================
    # State Accessors
    # ============================================================================

    @property
    def particle_positions(self) -> np.ndarray:
        """Get current particle positions from simulation state."""
        return self.state_0.particle_q.numpy()  # type: ignore

    @property
    def velocity(self) -> np.ndarray:
        """Get current particle velocities from simulation state."""
        return self.state_0.particle_qd.numpy()  # type: ignore

    # ============================================================================
    # Physics Simulation (State → Solver → Updated State)
    # ============================================================================

    def simulate(self):
        """Execute one physics timestep following Newton's pattern:
        State → Solver → Updated State
        """
        # Prepare state for simulation
        self.state_0.clear_forces()
        self.model.particle_grid.build(self.state_0.particle_q, self.search_radius)
        # Store manual body transforms before solving
        # manual_body_transforms = self._get_manual_body_transforms()

        # Run solver (State → Solver → Updated State)
        contacts = self.model.collide(self.state_0)
        self.solver.step(
            self.state_0, self.state_1, self.control, contacts, self.frame_dt
        )

        # Restore manual bodies (kinematic constraint)
        # self._restore_manual_body_transforms(manual_body_transforms)

        # Swap double-buffered states
        self.state_0, self.state_1 = self.state_1, self.state_0

    # def _get_manual_body_transforms(self) -> dict[int, np.ndarray]:
    #     """Get transforms of manually-controlled bodies before physics solve."""
    #     manual_transforms = {}
    #     if self.state_0.body_q is not None:
    #         current_transforms = self.state_0.body_q.numpy()
    #         for i, obj in enumerate(self.objects):
    #             if not obj.wb.is_active:
    #                 manual_transforms[i] = current_transforms[i].copy()
    #     return manual_transforms

    # def _restore_manual_body_transforms(self, manual_transforms: dict[int, np.ndarray]):
    #     """Restore manually-controlled bodies to fixed positions (kinematic behavior)."""
    #     if not manual_transforms or self.state_1.body_q is None:
    #         return

    #     # Restore positions
    #     solved_transforms = self.state_1.body_q.numpy()
    #     for i, transform in manual_transforms.items():
    #         solved_transforms[i] = transform
    #     self.state_1.body_q.assign(solved_transforms)

    #     # Zero velocities to prevent movement
    #     if hasattr(self.state_1, "body_qd") and self.state_1.body_qd is not None:
    #         body_velocities = self.state_1.body_qd.numpy()
    #         for i in manual_transforms.keys():
    #             body_velocities[i] = 0.0
    #         self.state_1.body_qd.assign(body_velocities)

    # ============================================================================
    # Visualization
    # ============================================================================

    def create_pointcloud(self):
        """Create or update Blender mesh for particle visualization."""
        name = "ParticleObject"

        self.particle_object = db.BlenderObject.from_pointcloud(
            self.particle_positions,
            name=name,
        )

        self.particle_object.store_named_attribute(
            np.array(self.builder.particle_radius),
            "radius",
        )

    def _update_particle_visualization(self):
        """Update particle mesh with current simulation state."""
        self.particle_object.position = self.particle_positions
        self.particle_object.store_named_attribute(self.velocity, "velocity")

    # ============================================================================
    # Main Simulation Loop
    # ============================================================================

    def step(self):
        """Execute one complete simulation step.

        Called whenever the Blender scene frame will change.

        Flow: Blender → Simulation → Solve → Blender
        """
        self._update_simulation_from_blender()
        start_simulate = time.time()
        self.simulate()
        self.props.time_compute = time.time() - start_simulate
        start_sync = time.time()
        self._update_blender_from_simulation()
        self._update_particle_visualization()
        self.props.time_sync = time.time() - start_sync
        self.clock += 1
