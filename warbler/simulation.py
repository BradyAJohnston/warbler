from .geometryset import GeometrySet
import bpy
import warp as wp
import newton
import numpy as np
import databpy as db
from .utils import smooth_lerp, blender_to_quat, quat_to_blender
from .props import WarblerSceneProperties, WarblerObjectProperties
from uuid import uuid1
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import SimulationManager


class Simulation:
    """Physics simulation integrating Newton Physics with Blender.

    Follows Newton's architecture pattern:
    ModelBuilder → Model → State → Solver → Updated State
    """

    # ============================================================================
    # Scene Properties (read from Blender)
    # ============================================================================

    @property
    def scene(self) -> bpy.types.Scene:
        return bpy.context.scene

    @property
    def props(self) -> WarblerSceneProperties:
        return self.scene.wb  # type: ignore

    @property
    def fps(self) -> int:
        return self.scene.render.fps

    @property
    def frame_dt(self) -> float:
        return 1 / self.fps

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
    def particle_radius(self) -> float:
        return self.props.particle_radius

    @property
    def uuid(self) -> str:
        return self._uuid

    @property
    def manager(self) -> "SimulationManager":
        if self._manager is None:
            raise RuntimeError
        return self._manager

    @property
    def is_active(self) -> bool:
        return self.manager.sim_items[self.uuid].is_active

    # ============================================================================
    # Initialization
    # ============================================================================

    def __init__(
        self,
        num_particles: int,
        substeps: int = 5,
        objects: list[bpy.types.Object] = [],
        up_vector=(0, 0, 1),
        ivelocity=(0, 0, 10),
        add_ground: bool = True,
        particle_object: bpy.types.Object | None = None,
    ):
        """Initialize simulation with specified parameters.

        Args:
            num_particles: Number of particles (will be cubed)
            substeps: Number of solver iterations per timestep
            links: Whether to add springs between particles
            objects: Blender objects to include as rigid bodies
            up_vector: World up direction
            ivelocity: Initial particle velocity
            add_ground: Whether to add ground plane
        """
        # Store configuration
        self.num_particles: int = num_particles
        self.objects: list = objects
        self.builder: newton.ModelBuilder = self._create_model_builder(up_vector)
        self.clock: int = 0
        self.bob: db.BlenderObject | None = None
        self._uuid: str = str(uuid1())
        self._manager: SimulationManager | None = None

        self._add_rigid_bodies(objects)
        if add_ground:
            self.builder.add_ground_plane()

        if particle_object is not None:
            geo = GeometrySet(particle_object)
            self._add_particles(**geo.pointcloud.to_props())
        else:
            positions = np.random.random((num_particles, 3))
            self._add_particles(positions)

        # Finalize model (Model phase)
        self.model = self.builder.finalize(device="cuda")

        # Create simulation state (State phase)
        self.state_0: newton.State = self.model.state()
        self.state_1: newton.State = self.model.state()

        # Create solver (Solver phase)
        self.solver = newton.solvers.SolverXPBD(
            model=self.model,
            iterations=substeps,
        )

        # Create control
        self.control = self.model.control()
        self.create_pointcloud()

    # ============================================================================
    # Model Building (ModelBuilder → Model)
    # ============================================================================

    def _create_model_builder(self, up_vector: tuple) -> newton.ModelBuilder:
        """Create and configure ModelBuilder with up axis and particle settings."""
        # Convert up_vector to Newton axis enum
        axis_map = {
            (0, 0, 1): newton.Axis.Z,
            (0, 1, 0): newton.Axis.Y,
            (1, 0, 0): newton.Axis.X,
        }
        up_axis = axis_map.get(up_vector, newton.Axis.Z)

        builder = newton.ModelBuilder(up_axis=up_axis)
        builder.default_particle_radius = self.particle_radius
        return builder

    def _add_rigid_bodies(self, objects: list[bpy.types.Object]):
        """Add Blender objects as rigid bodies to the model."""
        for obj in objects:
            obj.rotation_mode = "QUATERNION"

            # Create body with transform from Blender
            initial_transform = wp.transform(
                wp.vec3(*obj.location),
                wp.quat(*blender_to_quat(obj.rotation_quaternion)),
            )
            body = self.builder.add_body(mass=1e5, xform=initial_transform)

            # Valid ShapeConfig parameters (from newton.ModelBuilder.ShapeConfig signature)
            # Note: rolling_friction and torsional_friction are not ShapeConfig parameters

            props: list[str] = [
                x for x in dir(WarblerObjectProperties) if x.startswith("rigid_")
            ]

            # Source properties from Blender object and pass to ShapeConfig
            if obj.wb.rigid_shape == "CUBE":  # type: ignore
                prop_dict = {
                    name.replace("rigid_", ""): getattr(obj.wb, name) for name in props
                }  # type: ignore
                shape_cfg = newton.ModelBuilder.ShapeConfig(**prop_dict)

                obj.wb.rigid_body_index = self.builder.add_shape_box(  # type: ignore
                    body=body,
                    xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity(float)),
                    hx=obj.dimensions[0] / 2.0,
                    hy=obj.dimensions[1] / 2.0,
                    hz=obj.dimensions[2] / 2.0,
                    cfg=shape_cfg,
                )
            else:
                print(Warning(f"Unsupported shape {obj.wb.rigid_shape}"))  # type: ignore

    def _add_particles(
        self,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
        mass: np.ndarray | None = None,
        radius: np.ndarray | None = None,
    ) -> None:
        self.particle_radii = radius

        if velocity is None:
            velocity = np.zeros(position.shape, dtype=float)
        if mass is None:
            mass = np.repeat(1.0, position.shape[0])
        if radius is None:
            radius = np.repeat(0.1, position.shape[0])

        self.builder.add_particles(
            pos=position,  # type: ignore
            vel=velocity,  # type: ignore
            mass=mass,  # type: ignore
            radius=radius,  # type: ignore
        )

    def _add_particle_grid(
        self,
        num_particles: int,
        ivelocity: tuple,
    ):
        """Add particle grid to the model."""
        # Calculate grid dimensions
        n_x = n_y = n_z = int(num_particles ** (1 / 3))

        self.builder.add_particle_grid(
            dim_x=n_x,
            dim_y=n_y,
            dim_z=n_z,
            cell_x=0.1 * 2.0,
            cell_y=0.1 * 2.0,
            cell_z=0.1 * 2.0,
            pos=wp.vec3(-1.0, 0.0, 0.0),  # type: ignore
            rot=wp.quat_identity(),  # type: ignore
            vel=wp.vec3(*ivelocity),  # type: ignore
            mass=1,
            jitter=self.particle_radius * 0.1,
            radius_mean=self.particle_radius,
        )

        # Update actual particle count
        self.num_particles = n_x * n_y * n_z

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
            if obj.wb.rigid_is_active:
                obj.location = rigid_transforms[i, 0:3]
                obj.rotation_quaternion = quat_to_blender(rigid_transforms[i, 3:7])

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
            obj.rotation_mode = "QUATERNION"

            if not obj.wb.rigid_is_active:
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
        rot = blender_to_quat(obj.rotation_quaternion)
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
        if self.model.particle_grid is not None:
            self.model.particle_grid.build(
                self.state_0.particle_q, self.particle_radius * 2
            )  # type: ignore

        # Store manual body transforms before solving
        manual_body_transforms = self._get_manual_body_transforms()

        # Run solver (State → Solver → Updated State)
        contacts = self.model.collide(self.state_0)
        self.solver.step(
            self.state_0, self.state_1, self.control, contacts, self.frame_dt
        )

        # Restore manual bodies (kinematic constraint)
        self._restore_manual_body_transforms(manual_body_transforms)

        # Swap double-buffered states
        self.state_0, self.state_1 = self.state_1, self.state_0

    def _get_manual_body_transforms(self) -> dict[int, np.ndarray]:
        """Get transforms of manually-controlled bodies before physics solve."""
        manual_transforms = {}
        if self.state_0.body_q is not None:
            current_transforms = self.state_0.body_q.numpy()
            for i, obj in enumerate(self.objects):
                if not obj.wb.rigid_is_active:
                    manual_transforms[i] = current_transforms[i].copy()
        return manual_transforms

    def _restore_manual_body_transforms(self, manual_transforms: dict[int, np.ndarray]):
        """Restore manually-controlled bodies to fixed positions (kinematic behavior)."""
        if not manual_transforms or self.state_1.body_q is None:
            return

        # Restore positions
        solved_transforms = self.state_1.body_q.numpy()
        for i, transform in manual_transforms.items():
            solved_transforms[i] = transform
        self.state_1.body_q.assign(solved_transforms)

        # Zero velocities to prevent movement
        if hasattr(self.state_1, "body_qd") and self.state_1.body_qd is not None:
            body_velocities = self.state_1.body_qd.numpy()
            for i in manual_transforms.keys():
                body_velocities[i] = 0.0
            self.state_1.body_qd.assign(body_velocities)

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

        self.particle_object["radius"] = self.particle_radii

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
        self.simulate()
        self._update_blender_from_simulation()
        self._update_particle_visualization()

        self.clock += 1
