import bpy
import warp as wp
import newton
import numpy as np
import databpy as db
from .utils import smooth_lerp, blender_to_quat, quat_to_blender


class Simulation:
    def __init__(
        self,
        num_particles: int,
        substeps: int = 5,
        links: bool = False,
        objects: list[bpy.types.Object] = [],
        up_vector=(0, 0, 1),
        ivelocity=(0, 0, 10),
        add_ground: bool = True,
    ):
        # Initialize simulation parameters
        self.num_particles: int = num_particles
        self.radius: float = 0.1
        self.scale: float = 1.0
        self.objects: list = objects
        self.clock: int = 0
        self.fps: int = bpy.context.scene.render.fps
        self.frame_dt: float = 1 / self.fps
        self.ke: float = 1.0e5
        self.kd: float = 250.0
        self.kf: float = 500.0
        self.bob: db.BlenderObject | None = None

        # Create builder for simulation
        # Convert up_vector to up_axis enum
        if up_vector == (0, 0, 1):
            up_axis = newton.Axis.Z
        elif up_vector == (0, 1, 0):
            up_axis = newton.Axis.Y
        elif up_vector == (1, 0, 0):
            up_axis = newton.Axis.X
        else:
            # Default to Z if non-standard
            up_axis = newton.Axis.Z

        builder = newton.ModelBuilder(up_axis=up_axis)
        builder.default_particle_radius = self.radius

        n_x = int(num_particles ** (1 / 3))
        n_y = int(num_particles ** (1 / 3))
        n_z = int(num_particles ** (1 / 3))

        for obj in objects:
            # obj = bob.object
            # Set rotation mode to quaternion for consistent handling
            obj.rotation_mode = "QUATERNION"

            # Create body with initial transform from Blender object
            initial_transform = wp.transform(
                wp.vec3(*obj.location),
                wp.quat(*blender_to_quat(obj.rotation_quaternion)),
            )

            b = builder.add_body(
                mass=1e5,
                xform=initial_transform,
            )

            if obj.wb.rigid_shape == "CUBE":  # type: ignore
                # Create shape configuration with material properties
                shape_cfg = newton.ModelBuilder.ShapeConfig(
                    ke=self.ke,
                    kd=self.kd,
                    kf=self.kf,
                    ka=0.1,
                    mu=1.0,
                    restitution=100,
                )

                # Use object dimensions (which includes scale) for half-extents
                # obj.dimensions gives actual size in world space
                obj.wb.rigid_body_index = builder.add_shape_box(  # type: ignore
                    body=b,
                    xform=wp.transform(
                        wp.vec3(0, 0, 0), wp.quat_identity()
                    ),  # Shape centered on body
                    hx=obj.dimensions[0] / 2.0,  # Half-extents
                    hy=obj.dimensions[1] / 2.0,
                    hz=obj.dimensions[2] / 2.0,
                    cfg=shape_cfg,
                )
            else:
                print(Warning(f"Unsupported shape {obj.wb.rigid_shape}"))  # type: ignore

        # Add ground plane if requested
        if add_ground:
            builder.add_ground_plane()

        builder.add_particle_grid(
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
            jitter=self.radius * 0.1,
            radius_mean=self.radius,
        )

        self.num_particles = n_x * n_y * n_z
        self.edges = []
        if links:
            self.edges = np.zeros((self.num_particles, 2), int)
            for i in range(self.num_particles):
                self.edges[i, :] = (i - 1, i)
                builder.add_spring(i - 1, i, 3, 0.0, 0)

        # Finalize and build the model
        self.model = builder.finalize(device="cuda", requires_grad=False)

        # Create states
        self.state_0: newton.State = self.model.state()
        self.state_1: newton.State = self.model.state()
        self._state_initial = self.model.state()

        # Create solver
        self.solver = newton.solvers.SolverXPBD(
            model=self.model,
            iterations=substeps,
        )

        # Create control object
        self.control = self.model.control()

        # Create mesh object for visualization
        self.create_particle_mesh()

    def set_obj_from_simulation(self):
        if self.state_0.body_q is None:
            return None
        self.rigid_transforms = self.state_0.body_q.numpy()
        for i, obj in enumerate(self.objects):
            if not obj.wb.rigid_is_active:
                continue
            obj.location = self.rigid_transforms[i, 0:3]
            obj.rotation_quaternion = quat_to_blender(self.rigid_transforms[i, 3:7])

    def set_simulation_from_obj(self, decay_frames=5):
        new_transforms = []
        if self.state_0.body_q is None:
            return None
        current_sim_transforms = self.state_0.body_q.numpy()

        if self.state_0.body_q is None:
            return None

        # Get body velocities for manually controlled bodies
        body_velocities = None
        if hasattr(self.state_0, "body_qd") and self.state_0.body_qd is not None:
            body_velocities = self.state_0.body_qd.numpy()

        for i, obj in enumerate(self.objects):
            obj.rotation_mode = "QUATERNION"
            loc_in_sim = current_sim_transforms[i, 0:3]
            rot_in_sim = current_sim_transforms[i, 3:7]

            if not obj.wb.rigid_is_active:
                # Manually controlled body - set position from Blender with smoothing
                rot = blender_to_quat(obj.rotation_quaternion)
                loc = np.array(obj.location)

                # Apply smoothing to position (except first frame)
                if self.clock != 0:
                    loc = smooth_lerp(
                        loc_in_sim,
                        loc,
                        bpy.context.scene.wb.rigid_decay_frames,  # type: ignore
                    )

                    # Calculate velocity from smoothed position change for particle interaction
                    # This velocity will be used during collision resolution to push particles
                    # but will be zeroed after the solve to prevent the body from moving
                    if body_velocities is not None:
                        velocity = (loc - loc_in_sim) / self.frame_dt
                        body_velocities[i, 0:3] = velocity  # Linear velocity
                        # Angular velocity remains zero

                trans = wp.transform(wp.vec3(*loc), wp.quat(*rot))
            else:
                # Physics-controlled body - use simulation position
                trans = wp.transform(
                    wp.vec3(*loc_in_sim),
                    wp.quat(*rot_in_sim),
                )

            new_transforms.append(trans)

        self.state_0.body_q.assign(new_transforms)

        # Update body velocities if they were modified
        if body_velocities is not None:
            self.state_0.body_qd.assign(body_velocities)

    def create_particle_mesh(self):
        name = "ParticleObject"
        try:
            obj = bpy.data.objects[name]
            if len(obj.data.vertices) != self.num_particles:  # type: ignore
                bpy.data.objects.remove(obj)
                obj = db.create_object(
                    self.particle_positions, edges=self.edges, name=name
                )

            self.particle_obj = db.BlenderObject(obj)

            self.particle_obj.position = self.particle_positions
        except KeyError:
            self.particle_obj = db.create_bob(
                self.particle_positions,
                edges=self.edges,  # type: ignore
                name=name,
            )

        self.particle_obj.store_named_attribute(
            np.repeat(self.radius, self.num_particles), "radius"
        )

    @property
    def particle_positions(self) -> np.ndarray:
        return self.state_0.particle_q.numpy()  # type: ignore

    @property
    def velocity(self) -> np.ndarray:
        return self.state_0.particle_qd.numpy()  # type: ignore

    def simulate(self):
        self.state_0.clear_forces()
        # self.state_1.clear_forces()
        if self.model.particle_grid is not None:
            self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2)  # type: ignore

        # Store positions and rotations of manually controlled bodies before solving
        manual_body_transforms = {}
        if self.state_0.body_q is not None:
            current_transforms = self.state_0.body_q.numpy()
            for i, obj in enumerate(self.objects):
                if not obj.wb.rigid_is_active:
                    # Store the transform we want to keep
                    manual_body_transforms[i] = current_transforms[i].copy()

        contacts = self.model.collide(self.state_0)
        self.solver.step(
            self.state_0, self.state_1, self.control, contacts, self.frame_dt
        )

        # Restore manually controlled bodies to their intended positions (make them kinematic)
        if manual_body_transforms and self.state_1.body_q is not None:
            solved_transforms = self.state_1.body_q.numpy()
            for i, transform in manual_body_transforms.items():
                # Restore the position/rotation we set (ignore solver's changes)
                solved_transforms[i] = transform
            self.state_1.body_q.assign(solved_transforms)

            # Set their velocities to zero to prevent internal bouncing
            if hasattr(self.state_1, "body_qd") and self.state_1.body_qd is not None:
                body_velocities = self.state_1.body_qd.numpy()
                for i in manual_body_transforms.keys():
                    body_velocities[i] = 0.0  # Zero out all velocity components
                self.state_1.body_qd.assign(body_velocities)

        # swap states
        (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def object_as_wp_transform(self, obj: bpy.types.Object):
        return wp.transform(
            wp.vec3(obj.location), wp.quat(*blender_to_quat(obj.rotation_quaternion))
        )

    def step(self):
        # get the blender object the user is interacting with and extract
        # the transformations from it

        self.set_simulation_from_obj()
        self.simulate()
        self.set_obj_from_simulation()

        # Update particle mesh
        self.particle_obj.position = self.particle_positions
        self.particle_obj.store_named_attribute(self.velocity, "velocity")
        self.clock += 1
