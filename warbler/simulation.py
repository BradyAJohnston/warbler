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

from . import kernels, rigid
from .geometryset import GeometryAttributes, GeometrySet
from .props import SimulationListItem
from .utils import (
    angular_velocity,
    get_scene,
    world_transform,
    wp_transform,
)

if TYPE_CHECKING:
    from .manager import SimulationManager

# Opt in to the coordinate-layout joint_target_q (the future default). We never
# write joint targets, so this only silences the legacy-layout DeprecationWarning
# emitted by finalize() for models with free joints.
newton.use_coord_layout_targets = True

# A particle is colliding when within this fraction of its radius of a shape
CONTACT_TOLERANCE = 0.1


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
    faces, otherwise particles joined by springs along its edges).
    """

    component: str
    start: int
    count: int
    curve_sizes: np.ndarray | None = None
    faces: list[list[int]] = field(default_factory=list)
    # (N, 2) local particle indices joined by springs: the first
    # ``bond_count`` are mesh edges (permanent bonds), the rest network contacts
    edges: np.ndarray | None = None
    bond_count: int = 0
    # Index of this part's first spring in the model
    spring_start: int = 0
    # Source POINT attributes copied onto the output: name -> (values, type)
    attributes: dict[str, tuple[np.ndarray, str]] = field(default_factory=dict)
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
        if self.is_cloth or (self.edges is not None and len(self.edges) > 0):
            return "MESH"
        return "POINTCLOUD"


def network_edges(position: np.ndarray, cutoff: float) -> np.ndarray:
    """(N, 2) index pairs ``i < j`` of every two points closer than ``cutoff``.

    Brute force in blocks of rows, which is plenty for the few thousand beads
    of a coarse-grained protein.
    """
    position = np.asarray(position, dtype=np.float64)
    sq = np.einsum("ij,ij->i", position, position)
    pairs = [np.zeros((0, 2), dtype=np.int64)]
    for start in range(0, len(position), 256):
        block = position[start : start + 256]
        d2 = sq[start : start + 256, None] + sq[None, :] - 2.0 * block @ position.T
        i, j = np.nonzero(d2 < cutoff * cutoff)
        i += start
        keep = i < j
        pairs.append(np.column_stack([i[keep], j[keep]]))
    return np.concatenate(pairs).astype(np.int32)


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
        # (object, shape index) of each rigid box, for live scale updates
        self._rigid_shapes: list[tuple[Object, int]] = []
        # Advanced every substep so the thermostat draws fresh noise
        self._noise_seed: int = 0
        # Network contacts (springs that can break), set up in finalize()
        self._contact_ids: wp.array | None = None
        self._contact_ke: wp.array | None = None
        self._contact_broken: wp.array | None = None
        self._spring_rest: np.ndarray | None = None
        # (start, target) shape_scale to ramp between over this frame's substeps
        self._scale_ramp: tuple[np.ndarray, np.ndarray] | None = None
        # Particles held on props.surface, set up in finalize()
        self._surface_ids: list[np.ndarray] = []
        self._surface_offsets: list[np.ndarray] = []
        self._surface_mesh: wp.Mesh | None = None
        self._surface_indices: wp.array | None = None
        # Particles with ``attach``, joined to their nearest rigid body
        self._attach_ids: list[np.ndarray] = []
        # Solver contact forces (VBD only), set up in finalize()
        self._observables = None

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
            "_contact_ids",
            "_contact_ke",
            "_contact_broken",
            "_surface_points",
            "_surface_vertex_normals",
            "_surface_offset",
            "_surface_normal",
            "_particle_contact_f",
            "_particle_colliding",
            "_body_contact_f",
        ):
            if hasattr(self, name):
                delattr(self, name)
        self._contact_ids = None
        self._surface_mesh = None
        self._surface_indices = None
        self._observables = None

    def free(self) -> None:
        """Stop the simulation and release it, keeping the output objects.

        Settings that are fixed at compile time (solver, device, iterations)
        can then be changed before compiling again.
        """
        self._free_model()
        self.clock = 0
        self.props.is_compiled = False

    def build(self):
        # Rigid bodies read matrix_world, which is stale after a script edits
        # location / rotation / scale until the view layer is re-evaluated.
        if bpy.context.view_layer is not None:
            bpy.context.view_layer.update()
        self.builder = newton.ModelBuilder(
            up_axis=newton.Axis.Z, gravity=(0.0, 0.0, self.props.gravity)
        )
        self.builder.default_particle_radius = DEFAULT_RADIUS
        self.parts = []
        self._pinned_indices = None
        self._surface_ids: list[np.ndarray] = []
        self._surface_offsets: list[np.ndarray] = []
        self._rigid_shapes = []
        self._attach_ids = []

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

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        # DAT snapshots poses at the solver's own detection instants, so the
        # solver runs collision itself and step() gets contacts=None.
        self._solver_collides = self.props.solver == "VBD" and self.props.rigid_soft_dat
        if self.props.solver == "VBD":
            self.solver = newton.solvers.SolverVBD(
                model=self.model,
                iterations=self.props.iterations,
                rigid_compliant_alm=True,
                rigid_soft_contact_use_log_barrier=self.props.rigid_soft_log_barrier,
                rigid_soft_enable_dat=self.props.rigid_soft_dat,
                collision_pipeline=(
                    self.collision_pipeline if self._solver_collides else None
                ),
            )
        else:
            self.solver = newton.solvers.SolverXPBD(
                model=self.model,
                iterations=self.props.iterations,
            )
        self.control = self.model.control()
        if self._solver_collides:
            assert self.solver.contacts is not None
            self.contacts: newton.Contacts = self.solver.contacts
        else:
            self.contacts = self.collision_pipeline.contacts()
        self._init_network_contacts()
        self._init_surface()
        self._init_contact_outputs()

    def _init_contact_outputs(self) -> None:
        """Per-particle / per-body contact arrays, filled on each frame's last substep.

        Only VBD reports particle-vs-shape contact forces (XPBD's ``contact_f``
        covers body-body contacts only), so XPBD gets ``is_colliding`` alone.
        """
        device = self.model.device
        self._particle_contact_f = wp.zeros(
            self.model.particle_count, dtype=wp.vec3, device=device
        )
        self._particle_colliding = wp.zeros(
            self.model.particle_count, dtype=wp.int32, device=device
        )
        self._body_contact_f = wp.zeros(
            max(self.model.body_count, 1), dtype=wp.vec3, device=device
        )
        kind = self.solver.ObservableKind.CONTACT_F
        if self.props.solver == "VBD" and kind in self.solver.supported_observables:
            self._observables = self.solver.observables(
                kinds={kind}, contacts=self.contacts
            )

    @property
    def reports_contact_force(self) -> bool:
        return self._observables is not None

    def _gather_contacts(self) -> None:
        """Mark colliding particles and sum contact forces from the last substep."""
        if self.model.particle_count == 0:
            return
        self._particle_contact_f.zero_()
        self._particle_colliding.zero_()
        self._body_contact_f.zero_()
        if self._observables is not None:
            contact_f = self._observables.contact_f
        else:
            contact_f = wp.empty(0, dtype=wp.spatial_vector, device=self.model.device)
        body_q = self.state_0.body_q
        if body_q is None:
            body_q = wp.empty(0, dtype=wp.transform, device=self.model.device)
        c = self.contacts
        wp.launch(
            kernels.gather_soft_contacts,
            dim=c.soft_contact_max,
            inputs=[
                c.soft_contact_count,
                c.soft_contact_indices,
                c.soft_contact_barycentric,
                c.soft_contact_shape,
                c.soft_contact_body_pos,
                c.soft_contact_normal,
                self.model.shape_body,
                body_q,
                self.state_0.particle_q,
                self.model.particle_radius,
                contact_f,
                c.rigid_contact_max,
                CONTACT_TOLERANCE,
            ],
            outputs=[
                self._particle_contact_f,
                self._particle_colliding,
                self._body_contact_f,
            ],
            device=self.model.device,
        )

    def _init_surface(self) -> None:
        """Upload the particles held on ``props.surface`` and build its mesh."""
        self._surface_mesh = None
        self._surface_indices = None
        ids = np.concatenate(self._surface_ids) if self._surface_ids else []
        if len(ids) == 0:
            return
        device = self.model.device
        self._surface_indices = wp.array(ids, dtype=wp.int32, device=device)
        self._surface_offset = wp.array(
            np.concatenate(self._surface_offsets), dtype=float, device=device
        )
        self._surface_normal = wp.zeros(len(ids), dtype=wp.vec3, device=device)
        self._update_surface()
        self._project_to_surface()

    def _update_surface(self) -> None:
        """Rebuild the surface mesh from the evaluated ``props.surface``.

        Rebuilt rather than refit every frame, as remeshed surfaces (e.g. an
        SDF union as two vesicles fuse) change topology.
        """
        surface = self.props.surface
        if self._surface_indices is None or surface is None:
            return
        attrs = GeometrySet(surface).mesh
        mesh = attrs._data
        triangles = attrs.triangles
        if not isinstance(mesh, bpy.types.Mesh) or len(triangles) == 0:
            return
        mat = np.array(surface.matrix_world, dtype=np.float64)
        position = db.Attribute(mesh.attributes["position"]).as_array()
        position = position @ mat[:3, :3].T + mat[:3, 3]
        normals = np.zeros(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertex_normals.foreach_get("vector", normals)
        normals = normals.reshape(-1, 3) @ np.linalg.inv(mat[:3, :3])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        device = self.model.device
        self._surface_points = wp.array(position, dtype=wp.vec3, device=device)
        self._surface_vertex_normals = wp.array(normals, dtype=wp.vec3, device=device)
        self._surface_mesh = wp.Mesh(
            points=self._surface_points,
            indices=wp.array(triangles.ravel(), dtype=wp.int32, device=device),
        )

    def _project_to_surface(self) -> None:
        if self._surface_mesh is None or self._surface_indices is None:
            return
        wp.launch(
            kernels.project_to_surface,
            dim=len(self._surface_indices),
            inputs=[
                self._surface_mesh.id,
                self._surface_vertex_normals,
                self._surface_indices,
                self._surface_offset,
            ],
            outputs=[
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self._surface_normal,
            ],
            device=self.model.device,
        )

    def _init_network_contacts(self) -> None:
        """Gather the breakable network-contact springs of every part."""
        self._contact_ids = None
        self._spring_rest = None
        if self.model.spring_count == 0:
            return
        assert self.model.spring_rest_length is not None
        self._spring_rest = self.model.spring_rest_length.numpy()
        ids = [
            np.arange(p.spring_start + p.bond_count, p.spring_start + len(p.edges))
            for p in self.parts
            if p.edges is not None
        ]
        ids = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int32)
        if len(ids) == 0:
            return
        device = self.model.device
        assert self.model.spring_stiffness is not None
        native = self.model.spring_stiffness.numpy()[ids]
        self._contact_ids = wp.array(ids, dtype=wp.int32, device=device)
        self._contact_ke = wp.array(native, dtype=float, device=device)
        self._contact_broken = wp.zeros(len(ids), dtype=wp.int32, device=device)

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
                hx, hy, hz = rig.half_extents()
                shape = self.builder.add_shape_box(
                    body=body,
                    xform=wp_transform(),
                    hx=hx,
                    hy=hy,
                    hz=hz,
                    cfg=rig.shape_config(),
                )
                obj.wb.sim_body_index = shape  # type: ignore
                self._rigid_shapes.append((obj, shape))
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
            on_surface = props.pop("on_surface", None)
            surface_offset = props.pop("surface_offset", None)
            attach = props.pop("attach", None)
            part = SourcePart(
                component=component,
                start=self.builder.particle_count,
                count=len(props["position"]),
                attributes=attrs.point_attributes(),
                **kwargs,
            )
            if part.is_cloth:
                self._add_cloth(part, triangles=attrs.triangles, **props)
            else:
                self._add_particles(**props)
                if part.curve_sizes is not None:
                    self._add_curve_springs(part)
                else:
                    self._add_edge_springs(part, attrs, props["position"])
            if "pinned" in props:
                pinned.append(part.start + np.flatnonzero(props["pinned"]))
            if attach is not None:
                if "pinned" in props:
                    attach = attach & ~props["pinned"]
                self._attach_ids.append(part.start + np.flatnonzero(attach))
            if self.props.surface is not None:
                if on_surface is None:
                    on_surface = ~props.get("pinned", np.zeros(part.count, bool))
                local = np.flatnonzero(on_surface)
                self._surface_ids.append(part.start + local)
                self._surface_offsets.append(
                    np.zeros(len(local))
                    if surface_offset is None
                    else surface_offset[local]
                )
            self.parts.append(part)

        add("points", geo.points)
        add("curves", geo.curves, curve_sizes=geo.curves.curve_sizes)
        add("mesh", geo.mesh, faces=geo.mesh.faces)

        self._add_attachments()

        if pinned:
            self._pinned_indices = np.concatenate(pinned)
            for i in self._pinned_indices:
                # Zero mass AND non-ACTIVE so spring solves don't divide by zero
                self.builder.particle_mass[i] = 0.0
                self.builder.particle_flags[i] = 0

    def _add_attachments(self) -> None:
        """Join each ``attach`` particle to the nearest rigid body's surface.

        The body-local anchor is the particle's starting position, so the
        particle is carried along rigidly with the body. Only VBD solves
        attachments.
        """
        ids = np.concatenate(self._attach_ids) if self._attach_ids else []
        if len(ids) == 0 or not self.objects:
            return
        if self.props.solver != "VBD":
            print(Warning("Warbler: `attach` needs the VBD solver; ignored"))
            return
        boxes = []
        for obj in self.objects:
            rig = rigid.RigidObject(obj)
            inv = np.linalg.inv(np.array(obj.matrix_world.normalized()))
            boxes.append((inv, rig.half_extents()))
        position = np.array(self.builder.particle_q, dtype=float)[ids]
        homogeneous = np.c_[position, np.ones(len(ids))]
        local = np.stack([homogeneous @ inv.T for inv, _ in boxes])[..., :3]
        half = np.stack([h for _, h in boxes])[:, None, :]
        distance = np.linalg.norm(np.maximum(np.abs(local) - half, 0.0), axis=2)
        nearest = distance.argmin(axis=0)
        for k, (particle, body) in enumerate(zip(ids, nearest, strict=True)):
            self.builder.add_attachment_particle_body(
                int(particle),
                int(body),
                body_point=tuple(local[body, k]),
                stiffness=self.props.attach_stiffness,
                damping=self.props.attach_damping,
            )

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

    def _add_edge_springs(
        self, part: SourcePart, attrs: GeometryAttributes, position: np.ndarray
    ) -> None:
        """Join particles with bond springs along mesh edges and network contacts.

        Bonds (mesh edges) are permanent: spring stiffness/damping times the
        EDGE attributes ``bond_stiffness`` / ``bond_damping`` if present
        (distinct names from the POINT ``stiffness``, as a mesh can't hold
        two attributes of one name). With a network
        cutoff, every other pair of particles closer than it is joined by a
        contact (an elastic network model) of stiffness
        ``spring_ke × √(sᵢ·sⱼ)``, with ``s`` the POINT ``stiffness``
        attribute, scaled by ``domain_coupling`` unless both particles share a
        ``domain`` ≥ 0. Contacts can break and re-form (``break_strain``).
        Rest lengths are the starting distances.
        """
        ke, kd = self.props.spring_ke, self.props.spring_kd
        bonds = attrs.edges
        bond_k = attrs.edge_attribute("bond_stiffness")
        bond_d = attrs.edge_attribute("bond_damping")
        bond_k = ke * (np.ones(len(bonds)) if bond_k is None else bond_k)
        bond_d = kd * (np.ones(len(bonds)) if bond_d is None else bond_d)

        contacts = np.zeros((0, 2), dtype=np.int32)
        if self.props.network_cutoff > 0.0:
            contacts = network_edges(position, self.props.network_cutoff)
            existing = {tuple(sorted(e)) for e in bonds.tolist()}
            contacts = np.array(
                [e for e in contacts.tolist() if tuple(e) not in existing],
                dtype=np.int32,
            ).reshape(-1, 2)
        point_k = part.attributes.get("stiffness", (np.ones(part.count), ""))[0]
        domain = part.attributes.get("domain", (np.zeros(part.count), ""))[0]
        i, j = contacts[:, 0], contacts[:, 1]
        same_domain = (domain[i] == domain[j]) & (domain[i] >= 0)
        contact_k = (
            ke
            * np.sqrt(point_k[i] * point_k[j])
            * np.where(same_domain, 1.0, self.props.domain_coupling)
        )

        part.edges = np.concatenate([bonds, contacts]).astype(np.int32)
        part.bond_count = len(bonds)
        part.spring_start = self.builder.spring_count
        stiffness = np.concatenate([bond_k, contact_k])
        damping = np.concatenate([bond_d, np.full(len(contacts), kd)])
        for (a, b), k, d in zip(
            (part.edges + part.start).tolist(),
            stiffness.tolist(),
            damping.tolist(),
            strict=True,
        ):
            self.builder.add_spring(a, b, k, d, 0.0)

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
        forces = self._body_contact_f.numpy() if self.reports_contact_force else None
        for i, obj in enumerate(self.objects):
            rig = rigid.RigidObject(obj)
            if rig.is_active:
                rig.transform_from_wp(rigid_transforms[i])
            if forces is not None:
                rig.props.contact_force = forces[i]

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

    def _update_rigid_scales(self) -> None:
        """Queue a resize of rigid boxes to their objects' world-space size.

        The collision pipeline reads ``shape_scale`` (box half-extents) and
        ``shape_collision_radius`` (broadphase bounding sphere) from the model
        on every collide, so resizing takes effect without a re-compile. The
        change is ramped across the frame's substeps in ``simulate()``: applied
        in one substep, a growing box would bury particles and the contact
        solve would launch them. Mass and inertia are not updated, so for
        active bodies this only changes their collision size.
        """
        self._scale_ramp = None
        if not self._rigid_shapes or self.model.shape_scale is None:
            return
        start = self.model.shape_scale.numpy()
        target = start.copy()
        for obj, shape in self._rigid_shapes:
            target[shape] = rigid.RigidObject(obj).half_extents()
        if not np.allclose(start, target, rtol=1e-6, atol=1e-9):
            self._scale_ramp = (start, target)

    def _apply_rigid_scales(self, t: float) -> None:
        """Set rigid box sizes to fraction ``t`` of the way through the ramp."""
        if self._scale_ramp is None:
            return
        start, target = self._scale_ramp
        scale = start + (target - start) * t
        assert self.model.shape_scale is not None
        self.model.shape_scale.assign(scale)
        if self.model.shape_collision_radius is not None:
            radius = self.model.shape_collision_radius.numpy()
            shapes = [shape for _, shape in self._rigid_shapes]
            radius[shapes] = np.linalg.norm(scale[shapes], axis=1)
            self.model.shape_collision_radius.assign(radius)

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
        loc, rot = world_transform(obj)

        if self.clock != 0 and body_velocities is not None:
            prev_loc, prev_rot = current_sim_transform[0:3], current_sim_transform[3:7]
            body_velocities[body_index, 0:3] = (loc - prev_loc) / self.frame_dt
            body_velocities[body_index, 3:6] = angular_velocity(
                prev_rot, rot, self.frame_dt
            )

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

    def _update_network_contacts(self) -> None:
        """Break over-stretched network contacts and re-form relaxed ones."""
        if self._contact_ids is None:
            return
        wp.launch(
            kernels.update_network_contacts,
            dim=len(self._contact_ids),
            inputs=[
                self._contact_ids,
                self.model.spring_indices,
                self.model.spring_rest_length,
                self._contact_ke,
                self.state_0.particle_q,
                self.props.break_strain,
                self.props.reform_strain,
            ],
            outputs=[self._contact_broken, self.model.spring_stiffness],
            device=self.model.device,
        )

    @property
    def broken_springs(self) -> np.ndarray:
        """Boolean per model spring: a network contact that is currently broken."""
        broken = np.zeros(self.model.spring_count, dtype=bool)
        if self._contact_ids is not None and self._contact_broken is not None:
            broken[self._contact_ids.numpy()] = self._contact_broken.numpy() == 1
        return broken

    def _apply_thermostat(self, dt: float) -> None:
        """Langevin drag + thermal noise on particles (``temperature``/``drag``)."""
        drag, kT = self.props.drag, self.props.temperature
        if self.model.particle_count == 0 or drag <= 0.0:
            return
        self._noise_seed += 1
        wp.launch(
            kernels.langevin_forces,
            dim=self.model.particle_count,
            inputs=[
                self.state_0.particle_qd,
                self.model.particle_mass,
                self.model.particle_flags,
                drag,
                kT,
                dt,
                self._noise_seed,
            ],
            outputs=[self.state_0.particle_f],
            device=self.model.device,
        )

    def simulate(self):
        """Execute one physics timestep following Newton's pattern:
        State → Solver → Updated State
        """
        substeps = max(self.substeps, 1)
        sim_dt = self.frame_dt / substeps
        for i in range(substeps):
            self._apply_rigid_scales((i + 1) / substeps)
            self.state_0.clear_forces()
            self._apply_thermostat(sim_dt)
            last = i == substeps - 1
            if not self._solver_collides:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                None if self._solver_collides else self.contacts,
                sim_dt,
                # forces are only read back once a frame
                observables=self._observables if last else None,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._project_to_surface()
            self._update_network_contacts()
        self._gather_contacts()

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
            for attr_name, (values, data_type) in part.attributes.items():
                part.output.store_named_attribute(
                    values,
                    attr_name,
                    atype=data_type,  # ty: ignore[invalid-argument-type]
                )
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
        if part.object_type == "MESH":
            return db.BlenderObject.from_mesh(
                pos,
                edges=None if part.is_cloth else part.edges,
                faces=part.faces or None,
                name=name,
                collection=collection,
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
            bob.new_from_pydata(
                pos,
                edges=None if part.is_cloth else part.edges,
                faces=part.faces or None,
            )
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
        broken = self.broken_springs if self._contact_ids is not None else None
        colliding = self._particle_colliding.numpy().astype(bool)
        force = self._particle_contact_f.numpy() if self.reports_contact_force else None
        normal = None
        if self._surface_mesh is not None and self._surface_indices is not None:
            normal = np.zeros((self.model.particle_count, 3), dtype=np.float32)
            normal[self._surface_indices.numpy()] = self._surface_normal.numpy()
        for part in self.parts:
            if part.output is None:
                continue
            part.output.position = positions[part.indices]  # type: ignore[assignment]
            part.output.store_named_attribute(velocity[part.indices], "velocity")
            if normal is not None:
                part.output.store_named_attribute(
                    normal[part.indices], "surface_normal"
                )
            part.output.store_named_attribute(
                colliding[part.indices], "is_colliding", atype="BOOLEAN"
            )
            if force is not None:
                part.output.store_named_attribute(force[part.indices], "contact_force")
            if part.edges is not None and len(part.edges) and not part.is_cloth:
                self._update_edge_outputs(part, positions, broken)

    def _update_edge_outputs(
        self, part: SourcePart, positions: np.ndarray, broken: np.ndarray | None
    ) -> None:
        """Store per-edge ``strain`` and ``broken`` on a network output mesh.

        Output edges are in spring order, so GN can colour or hide them.
        """
        assert part.output is not None and part.edges is not None
        assert self._spring_rest is not None
        springs = slice(part.spring_start, part.spring_start + len(part.edges))
        q = positions[part.edges + part.start]
        length = np.linalg.norm(q[:, 0] - q[:, 1], axis=1)
        strain = length / self._spring_rest[springs] - 1.0
        part.output.store_named_attribute(
            strain.astype(np.float32), "strain", domain="EDGE"
        )
        if broken is not None:
            part.output.store_named_attribute(
                broken[springs], "broken", atype="BOOLEAN", domain="EDGE"
            )

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
            self._update_rigid_scales()
            self._update_pinned_particles()
            self._update_surface()
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
