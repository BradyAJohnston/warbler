import bpy
import warp as wp
from warp import sim
import numpy as np
from . import databpy as db


class Warbler:
    def __init__(
        self,
        num_particles: int,
        substeps: int = 5,
        links: bool = False,
        up_vector=(0, 0, 1),
        ivelocity=(0, 0, 10),
    ):
        # Initialize Warp
        wp.init()

        # Initialize simulation parameters
        self.num_particles = num_particles
        self.radius = 0.15
        self.frame_dt = 1.0 / 24  # 60 fps
        self.scale = 1

        # Create builder for simulation
        builder = sim.ModelBuilder(up_vector=wp.vec3(*up_vector))
        builder.default_particle_radius = self.radius

        n_x = int(num_particles ** (1 / 3))
        n_y = int(num_particles ** (1 / 3))
        n_z = int(num_particles ** (1 / 3))

        b = builder.add_body()

        builder.add_shape_box(
            pos=wp.vec3(0, 0, 0),
            hx=1 * self.scale,
            hy=1 * self.scale,
            hz=1 * self.scale,
            body=b,
        )

        builder.add_particle_grid(
            dim_x=n_x,
            dim_y=n_y,
            dim_z=n_z,
            cell_x=0.1 * 2.0,
            cell_y=0.1 * 2.0,
            cell_z=0.1 * 2.0,
            pos=wp.vec3(-1.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(*ivelocity),
            mass=1,
            jitter=self.radius * 0.1,
        )

        self.num_particles = n_x * n_y * n_z
        self.edges = []
        if links:
            self.edges = np.zeros((self.num_particles, 2), int)
            for i in range(self.num_particles):
                self.edges[i, :] = (i - 1, i)
                builder.add_spring(i - 1, i, 3, 0.0, 0)

        # Finalize and build the model
        self.model = builder.finalize("cuda")
        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Create integrator
        self.integrator = wp.sim.XPBDIntegrator(substeps)

        # Create mesh object for visualization
        self.create_particle_mesh()

    def create_particle_mesh(self):
        name = "ParticleObject"
        try:
            self.bob = db.BlenderObject(bpy.data.objects[name])
            self.bob.position = self.position
        except KeyError:
            self.bob = db.create_bob(self.position, edges=self.edges, name=name)

    @property
    def position(self) -> np.ndarray:
        return self.state_0.particle_q.numpy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state_0.particle_qd.numpy()

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2)
        wp.sim.collide(self.model, self.state_0)
        self.integrator.simulate(self.model, self.state_0, self.state_1, 1 / 24)
        # swap states
        (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def object_as_wp_transform(self, obj: bpy.types.Object):
        return wp.transform(wp.vec3(obj.location), wp.quat(obj.rotation_quaternion))

    def step(self):
        # get the blender object the user is interacting with and extract
        # the transformations from it
        try:
            obj = bpy.data.objects["Cube"]
            if obj.rotation_mode == "QUATERNION":
                rot = obj.rotation_quaternion
            elif obj.rotation_mode == "XYZ":
                rot = obj.rotation_euler.to_quaternion()
            else:
                raise ValueError(f"Unsupported rotation {obj.rotation_type}")

            transform = wp.transform(wp.vec3(*obj.location), wp.quat(*rot))
            self.state_0.body_q.assign([transform])
        except KeyError:
            pass
        # Run simulation substeps
        self.simulate()

        # Update Blender mesh
        self.bob.position = self.position
        self.bob.store_named_attribute(self.velocity, "velocity")
