import bpy
import warp as wp
from warp import sim
import numpy as np
import databpy as db
from .utils import smooth_lerp


class Simulation:
    def __init__(
        self,
        num_particles: int,
        substeps: int = 5,
        links: bool = False,
        objects: list[bpy.types.Object] = [],
        up_vector=(0, 0, 1),
        ivelocity=(0, 0, 10),
    ):
        # Initialize Warp
        wp.init()

        # Initialize simulation parameters
        self.num_particles: int = num_particles
        self.radius: float = 0.4
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
        builder = sim.ModelBuilder(up_vector=wp.vec3(*up_vector))
        builder.default_particle_radius = self.radius

        n_x = int(num_particles ** (1 / 3))
        n_y = int(num_particles ** (1 / 3))
        n_z = int(num_particles ** (1 / 3))

        for obj in objects:
            b = builder.add_body(m=1e5)
            if obj.wb.rigid_shape == "CUBE":
                obj.wb.rigid_body_index = builder.add_shape_box(
                    pos=wp.vec3(0, 0, 0),
                    hx=obj.scale[0],
                    hy=obj.scale[1],
                    hz=obj.scale[2],
                    kd=self.kd,
                    ke=self.ke,
                    kf=self.kf,
                    ka=0.1,
                    mu=1.0,
                    restitution=100,
                    body=b,
                )
            else:
                print(Warning(f"Unsupported shape {obj.wb.rigid_shape}"))

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
        self.state_0: sim.State = self.model.state()
        self.state_1: sim.State = self.model.state()
        self._state_initial = self.model.state()

        # Create integrator
        self.integrator = sim.XPBDIntegrator(substeps)

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
            obj.rotation_quaternion = self.rigid_transforms[i, 3:7]

    def set_simulation_from_obj(self, decay_frames=5):
        new_transforms = []
        if self.state_0.body_q is None:
            return None
        current_sim_transforms = self.state_0.body_q.numpy()

        if self.state_0.body_q is None:
            return None

        for i, obj in enumerate(self.objects):
            obj.rotation_mode = "QUATERNION"
            loc_in_sim = current_sim_transforms[i, 0:3]
            rot_in_sim = current_sim_transforms[i, 3:7]

            if not obj.wb.rigid_is_active or self.clock == 0:
                rot = np.array(obj.rotation_quaternion)
                loc = np.array(obj.location)
                if self.clock != 0:
                    loc = smooth_lerp(
                        loc_in_sim, loc, bpy.context.scene.wb.rigid_decay_frames
                    )

                trans = wp.transform(wp.vec3(*loc), wp.quat(*rot))
            else:
                trans = wp.transform(
                    wp.vec3(*loc_in_sim),
                    wp.quat(*rot_in_sim),
                )

            new_transforms.append(trans)

        self.state_0.body_q.assign(new_transforms)

    def create_particle_mesh(self):
        name = "ParticleObject"
        try:
            obj = bpy.data.objects[name]
            if len(obj.data.vertices) != self.num_particles:
                bpy.data.objects.remove(obj)
                obj = db.create_object(
                    self.particle_positions, edges=self.edges, name=name
                )

            self.particle_obj = db.BlenderObject(obj)

            self.particle_obj.position = self.particle_positions
        except KeyError:
            self.particle_obj = db.create_bob(
                self.particle_positions, edges=self.edges, name=name
            )

        self.particle_obj.store_named_attribute(
            np.repeat(self.radius, self.num_particles), "radius"
        )

    @property
    def particle_positions(self) -> np.ndarray:
        return self.state_0.particle_q.numpy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state_0.particle_qd.numpy()

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2)
        sim.collide(self.model, self.state_0)
        self.integrator.simulate(self.model, self.state_0, self.state_1, self.frame_dt)

        # swap states
        (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def object_as_wp_transform(self, obj: bpy.types.Object):
        return wp.transform(wp.vec3(obj.location), wp.quat(obj.rotation_quaternion))

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
