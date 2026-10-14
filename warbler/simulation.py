import time
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid1

import bpy
import databpy as db
import newton
import numpy as np
import warp as wp
from bpy.types import Object

from . import rigid
from .geometryset import GeometrySet
from .props import SimulationListItem
from .utils import (
    blender_rotation,
    get_scene,
    wp_transform,
)

if TYPE_CHECKING:
    from .manager import SimulationManager

# Opt in to the coordinate-layout joint_target_q (the future default). We never
# write joint targets, so this only silences the legacy-layout DeprecationWarning
# emitted by finalize() for models with free joints.
newton.use_coord_layout_targets = True


class SimulatorBase(ABC):
    @property
    def scene(self) -> bpy.types.Scene:
        return get_scene()

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


# Fallbacks when the source geometry has no `mass` / `radius` attribute. A mass
# of ~0.1 kg per particle matches Newton's cloth examples; much lighter
# particles make contact solves diverge (see CLAUDE.md).
DEFAULT_MASS = 0.1
DEFAULT_RADIUS = 0.05


@dataclass
class SourcePart:
    """A contiguous run of simulation particles read from one geometry component.

    ``component`` is the GeometrySet component the part was read from: ``points``
    (free particles), ``curves`` (spring chains) or ``mesh`` (cloth when it has
    faces, free particles otherwise).
    """

    component: str
    start: int
    count: int
    curve_sizes: np.ndarray | None = None
    faces: list[list[int]] = field(default_factory=list)
    output: db.BlenderObject | None = None

    @property
    def indices(self) -> slice:
        return slice(self.start, self.start + self.count)

    @property
    def is_cloth(self) -> bool:
        return self.component == "mesh" and len(self.faces) > 0

    @property
    def object_type(self) -> str:
        """Blender object type of this part's output."""
        if self.curve_sizes is not None:
            return "CURVES"
        return "MESH" if self.is_cloth else "POINTCLOUD"


class Simulator(SimulatorBase):
    """Physics simulation integrating Newton Physics with Blender.

    Follows Newton's architecture pattern:
    ModelBuilder → Model → State → Solver → Updated State

    The evaluated geometry of ``props.source`` is split by component: points
    become free particles, curves become spring chains and meshes with faces
    become cloth. Each part gets its own output object that is updated every
    frame.
    """

    @property
    def scale(self) -> float:
        return self.props.scale

    @property
    def objects(self) -> list[Object]:
        if self.props.sim_rigid_collection is None:
            return []
        return [o for o in self.props.sim_rigid_collection.objects]

    @property
    def outputs(self) -> list[db.BlenderObject]:
        return [p.output for p in self.parts if p.output is not None]

    # ============================================================================
    # Initialization
    # ============================================================================

    def __init__(
        self,
    ):
        """Initialize simulation with specified parameters."""
        super().__init__()
        self.clock: int = 0
        self.parts: list[SourcePart] = []
        self._pinned_indices: np.ndarray | None = None

    def _compile(self) -> None:
        self._free_model()
        previous = {p.component: p.output for p in self.parts if p.output is not None}
        self.build()
        self.finalize()
        self._create_outputs(previous)

    def _free_model(self) -> None:
        """Drop the Newton model, states and solver so their memory is released."""
        for name in (
            "builder",
            "model",
            "state_0",
            "state_1",
            "solver",
            "control",
            "collision_pipeline",
            "contacts",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def free(self) -> None:
        """Stop the simulation and release it, keeping the output objects.

        Settings that are fixed at compile time (solver, device, iterations)
        can then be changed before compiling again.
        """
        self._free_model()
        self.clock = 0
        self.props.is_compiled = False

    def build(self):
        self.builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        self.builder.default_particle_radius = DEFAULT_RADIUS
        self.parts = []
        self._pinned_indices = None

        self._add_rigid_bodies(self.objects)
        if self.props.use_ground_plane:
            self.builder.add_ground_plane()

        if self.props.source is not None:
            self._add_source(GeometrySet(self.props.source))

    def finalize(self):
        if self.props.solver == "VBD":
            # VBD solves particles in parallel by graph colour.
            self.builder.color(include_bending=True)
        self.model: newton.Model = self.builder.finalize(device=self.device)
        self.model.soft_contact_ke = self.props.soft_contact_ke
        self.model.soft_contact_kd = self.props.soft_contact_kd
        self.model.soft_contact_mu = self.props.soft_contact_mu

        self.state_0: newton.State = self.model.state()
        self.state_1: newton.State = self.model.state()

        if self.props.solver == "VBD":
            self.solver = newton.solvers.SolverVBD(
                model=self.model,
                iterations=self.props.iterations,
            )
        else:
            self.solver = newton.solvers.SolverXPBD(
                model=self.model,
                iterations=self.props.iterations,
            )
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

    def _add_rigid_bodies(self, objects: list[bpy.types.Object]):
        """Add Blender objects as rigid bodies to the model."""
        for obj in objects:
            rig = rigid.RigidObject(obj)
            body = self.builder.add_body(
                mass=1e5,
                xform=rig.wp_transform(),
                is_kinematic=not rig.is_active,
            )

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

    # ============================================================================
    # Source geometry → particles, springs and cloth
    # ============================================================================

    def _world_props(self, props: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Move position / velocity from source object space into world space."""
        source = self.props.source
        if source is None:
            return props
        mat = np.array(source.matrix_world, dtype=np.float64)
        out = dict(props)
        if "position" in out:
            out["position"] = out["position"] @ mat[:3, :3].T + mat[:3, 3]
        if "velocity" in out:
            out["velocity"] = out["velocity"] @ mat[:3, :3].T
        return out

    def _add_source(self, geo: GeometrySet) -> None:
        """Add every component of the evaluated source geometry to the builder."""
        pinned: list[np.ndarray] = []

        def add(component: str, attrs, **kwargs) -> None:
            props = self._world_props(attrs.to_props())
            if len(props.get("position", [])) == 0:
                return
            part = SourcePart(
                component=component,
                start=self.builder.particle_count,
                count=len(props["position"]),
                **kwargs,
            )
            if part.is_cloth:
                self._add_cloth(part, triangles=attrs.triangles, **props)
            else:
                self._add_particles(**props)
                if part.curve_sizes is not None:
                    self._add_curve_springs(part)
            if "pinned" in props:
                pinned.append(part.start + np.flatnonzero(props["pinned"]))
            self.parts.append(part)

        add("points", geo.points)
        add("curves", geo.curves, curve_sizes=geo.curves.curve_sizes)
        add("mesh", geo.mesh, faces=geo.mesh.faces)

        if pinned:
            self._pinned_indices = np.concatenate(pinned)
            for i in self._pinned_indices:
                # Zero mass AND non-ACTIVE so spring solves don't divide by zero
                self.builder.particle_mass[i] = 0.0
                self.builder.particle_flags[i] = 0

    def _add_particles(
        self,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
        mass: np.ndarray | None = None,
        radius: np.ndarray | None = None,
        pinned: np.ndarray | None = None,
    ) -> None:
        n = position.shape[0]
        if velocity is None:
            velocity = np.zeros(position.shape, dtype=float)
        if mass is None:
            mass = np.repeat(DEFAULT_MASS, n)
        if radius is None:
            radius = np.repeat(DEFAULT_RADIUS, n)

        self.builder.add_particles(
            pos=position,  # type: ignore
            vel=velocity,  # type: ignore
            mass=mass,  # type: ignore
            radius=radius,  # type: ignore
            flags=[int(newton.ParticleFlags.ACTIVE)] * n,
        )

    def _add_curve_springs(self, part: SourcePart) -> None:
        """Connect the points of each curve with stretch and bend springs.

        Stretch springs join neighbouring points; bend springs skip one point
        (i → i+2) and resist folding.
        """
        assert part.curve_sizes is not None
        ke, kd, bend_ke = self.props.spring_ke, self.props.spring_kd, self.props.bend_ke
        offsets = part.start + np.concatenate([[0], np.cumsum(part.curve_sizes)])
        for first, end in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True):
            for i in range(first, end - 1):
                self.builder.add_spring(i, i + 1, ke, kd, 0.0)
            if bend_ke > 0.0:
                for i in range(first, end - 2):
                    self.builder.add_spring(i, i + 2, bend_ke, kd, 0.0)

    def _add_cloth(
        self,
        part: SourcePart,
        triangles: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
        mass: np.ndarray | None = None,
        radius: np.ndarray | None = None,
        pinned: np.ndarray | None = None,
    ) -> None:
        """Add a triangulated mesh as cloth.

        VBD uses the triangle (membrane) and edge (bending) energies directly.
        XPBD has no triangle constraint, so stretch comes from springs along
        the mesh edges instead.
        """
        props = self.props
        self.builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),  # ty: ignore[invalid-argument-type]
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(*p) for p in position],
            indices=triangles.flatten().tolist(),
            density=1.0,  # overwritten by per-vertex mass below
            tri_ke=props.cloth_tri_ke,
            tri_ka=props.cloth_tri_ka,
            tri_kd=props.cloth_tri_kd,
            edge_ke=props.cloth_edge_ke,
            edge_kd=props.cloth_edge_kd,
            add_springs=props.solver == "XPBD",
            spring_ke=props.spring_ke,
            spring_kd=props.spring_kd,
            particle_radius=DEFAULT_RADIUS,
        )
        # add_cloth_mesh derives mass from density × triangle area, which is
        # tiny on fine meshes and makes contacts diverge. Use an explicit
        # per-vertex mass instead, like Newton's cloth examples.
        for local, i in enumerate(range(part.start, part.start + part.count)):
            self.builder.particle_mass[i] = (
                float(mass[local]) if mass is not None else DEFAULT_MASS
            )
            if radius is not None:
                self.builder.particle_radius[i] = float(radius[local])
            if velocity is not None:
                self.builder.particle_qd[i] = wp.vec3(*velocity[local])

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
                transform = self._get_kinematic_body_transform(
                    obj, current_sim_transforms[i], body_velocities, i
                )
            else:
                transform = wp.transform(
                    wp.vec3(*current_sim_transforms[i, 0:3]),
                    wp.quat(*current_sim_transforms[i, 3:7]),
                )

            new_transforms.append(transform)

        self.state_0.body_q.assign(new_transforms)
        if body_velocities is not None and self.state_0.body_qd is not None:
            self.state_0.body_qd.assign(body_velocities)

    def _update_pinned_particles(self) -> None:
        """Re-read pinned particle positions from the GN source each frame.

        Because particle_q_init is cloned from state_in.particle_q at the start
        of every solver.step(), writing updated positions here propagates
        animated pin targets into the simulation automatically.
        """
        if (
            self._pinned_indices is None
            or len(self._pinned_indices) == 0
            or self.props.source is None
            or self.state_0.particle_q is None
        ):
            return

        geo = GeometrySet(self.props.source)
        all_q = self.state_0.particle_q.numpy()
        for part in self.parts:
            attrs = getattr(geo, part.component)
            props = self._world_props(attrs.to_props())
            if "pinned" not in props or len(props["pinned"]) != part.count:
                # Topology changed since compile; pins can't be matched up
                continue
            local = np.flatnonzero(props["pinned"])
            all_q[part.start + local] = props["position"][local]
        self.state_0.particle_q.assign(all_q)

    def _get_kinematic_body_transform(
        self,
        obj: bpy.types.Object,
        current_sim_transform: np.ndarray,
        body_velocities: np.ndarray | None,
        body_index: int,
    ) -> wp.transform:
        """Return the transform for a kinematic (user-controlled) rigid body.

        The body is marked is_kinematic in Newton so the solver never applies
        forces to it. We still write velocity into body_qd so Newton can
        generate correct collision impulses on particles the body sweeps through.
        """
        rot = blender_rotation(obj.rotation_quaternion)
        loc = np.array(obj.location)

        if self.clock != 0 and body_velocities is not None:
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
        substeps = max(self.substeps, 1)
        sim_dt = self.frame_dt / substeps
        for _ in range(substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, sim_dt
            )
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

    def _output_collection(self) -> bpy.types.Collection | None:
        source = self.props.source
        if source is not None and source.users_collection:
            return source.users_collection[0]
        return None

    def _create_outputs(
        self, previous: dict[str, db.BlenderObject] | None = None
    ) -> None:
        """Create or reuse a Blender object for each source part's sim results.

        An existing output (from the previous compile, or found by name) keeps
        its object and data block and only has its geometry rebuilt, so
        modifiers and materials on it survive a re-compile. Outputs live at the
        world origin since simulation positions are in world space.
        """
        previous = dict(previous or {})
        source = self.props.source
        if self.parts:
            assert source is not None
            positions = self.particle_positions
            assert self.model.particle_radius is not None
            radius = self.model.particle_radius.numpy()
            pinned = np.zeros(self.model.particle_count, dtype=bool)
            if self._pinned_indices is not None:
                pinned[self._pinned_indices] = True

        for part in self.parts:
            assert source is not None
            name = f"{source.name}_{part.component}"
            pos = positions[part.indices]
            obj = previous.pop(part.component, None)
            obj = obj.object if obj is not None else bpy.data.objects.get(name)
            if obj is not None and obj.type == part.object_type:
                part.output = self._rebuild_output(obj, part, pos)
            else:
                part.output = self._new_output(name, part, pos)
            part.output.store_named_attribute(radius[part.indices], "radius")
            if self._pinned_indices is not None:
                part.output.store_named_attribute(pinned[part.indices], "pinned")

        # Parts that no longer exist in the source geometry
        for bob in previous.values():
            try:
                bpy.data.objects.remove(bob.object)
            except ReferenceError:
                pass  # already deleted by the user

    def _new_output(
        self, name: str, part: SourcePart, pos: np.ndarray
    ) -> db.BlenderObject:
        collection = self._output_collection()
        if part.curve_sizes is not None:
            return db.BlenderObject.from_curves(
                pos, part.curve_sizes, name=name, collection=collection
            )
        if part.is_cloth:
            return db.BlenderObject.from_mesh(
                pos, faces=part.faces, name=name, collection=collection
            )
        return db.BlenderObject.from_pointcloud(pos, name=name, collection=collection)

    @staticmethod
    def _rebuild_output(
        obj: bpy.types.Object, part: SourcePart, pos: np.ndarray
    ) -> db.BlenderObject:
        """Clear ``obj``'s geometry in place and refill it for ``part``."""
        bob = db.BlenderObject(obj)
        data = obj.data
        if isinstance(data, bpy.types.Mesh):
            bob.new_from_pydata(pos, faces=part.faces)
            return bob
        if isinstance(data, bpy.types.Curves):
            assert part.curve_sizes is not None
            data.remove_curves()
            data.add_curves(part.curve_sizes.tolist())
        elif isinstance(data, bpy.types.PointCloud):
            data.resize(0)
            data.resize(len(pos))
        bob.position = pos  # type: ignore[assignment]
        return bob

    def _update_outputs(self) -> None:
        """Write current particle positions and velocities to the output objects."""
        if self.model.particle_count == 0:
            return
        positions = self.particle_positions
        velocity = self.velocity
        for part in self.parts:
            if part.output is None:
                continue
            part.output.position = positions[part.indices]  # type: ignore[assignment]
            part.output.store_named_attribute(velocity[part.indices], "velocity")

    # ============================================================================
    # Main Simulation Loop
    # ============================================================================

    def _check_device(self) -> bool:
        """Return False and deactivate if the Warp device is in an error state."""
        try:
            wp.synchronize_device(self.device)
            return True
        except Exception as e:  # noqa: BLE001 — any device error means the context is unusable
            self.props.is_active = False
            print(
                f"Warbler: CUDA context is in an error state (likely from a previous "
                f"simulation crash). Restart Blender to recover. Detail: {e}"
            )
            return False

    def step(self):
        """Execute one complete simulation step.

        Called whenever the Blender scene frame will change.

        Flow: Blender → Simulation → Solve → Blender
        """
        if self.device == "cuda" and not self._check_device():
            return
        try:
            self._update_simulation_from_blender()
            self._update_pinned_particles()
            start_simulate = time.time()
            self.simulate()
            self.props.time_compute = time.time() - start_simulate
            start_sync = time.time()
            self._update_blender_from_simulation()
            self._update_outputs()
            self.props.time_sync = time.time() - start_sync
            self.clock += 1
        except Exception as e:  # noqa: BLE001
            # Warp CUDA errors (e.g. illegal memory access from an unstable
            # simulation) surface here as RuntimeError. Deactivate so Blender
            # doesn't keep hitting the same error every frame.
            self.props.is_active = False
            print(
                f"Warbler: simulation deactivated after error on frame {self.clock}: {e}\n"
                f"  Tip: increase substeps (current={self.substeps})."
            )
