"""Tests for cloth simulation via add_cloth_mesh."""

import bpy
import databpy as db
import numpy as np

from warbler.manager import get_manager
from warbler.simulation import SimulatorXPBD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_grid(nx: int, ny: int, spacing: float = 0.1) -> bpy.types.Object:
    """Create a flat (z=2) grid mesh suitable for cloth simulation."""
    verts = []
    for iy in range(ny):
        for ix in range(nx):
            verts.append((ix * spacing, iy * spacing, 2.0))

    faces = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a = iy * nx + ix
            b = a + 1
            c = a + nx + 1
            d = a + nx
            faces.append((a, b, c, d))

    bob = db.BlenderObject.from_mesh(
        vertices=np.array(verts, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
        name="ClothGrid",
    )
    return bob.object


def _compile_cloth_sim(
    cloth_obj: bpy.types.Object, device: str = "cpu"
) -> SimulatorXPBD:
    man = get_manager(bpy.context)
    sim = SimulatorXPBD()
    man.add(sim)
    item = man.active_item
    item.cloth_source = cloth_obj
    item.device = device
    item.use_ground_plane = True
    item.substeps = 3
    sim.compile()
    return sim


# ---------------------------------------------------------------------------
# Compilation tests
# ---------------------------------------------------------------------------


def test_cloth_compiles():
    """A cloth simulation should compile without error."""
    cloth_obj = _flat_grid(4, 4)
    sim = _compile_cloth_sim(cloth_obj)
    assert sim.props.is_compiled


def test_cloth_particle_count():
    """Cloth particle count should equal vertex count of the source mesh."""
    nx, ny = 5, 4
    cloth_obj = _flat_grid(nx, ny)
    sim = _compile_cloth_sim(cloth_obj)
    assert sim._cloth_particle_count == nx * ny


def test_cloth_particle_start_is_zero_without_particles():
    """When no separate particle source is set, cloth starts at index 0."""
    cloth_obj = _flat_grid(3, 3)
    sim = _compile_cloth_sim(cloth_obj)
    assert sim._cloth_particle_start == 0


def test_cloth_creates_output_object():
    """After compilation a WarblerCloth mesh object should exist in the scene."""
    cloth_obj = _flat_grid(4, 4)
    sim = _compile_cloth_sim(cloth_obj)
    assert sim.cloth_object is not None
    assert bpy.data.objects.get("WarblerCloth") is not None


def test_cloth_output_vertex_count_matches_source():
    cloth_obj = _flat_grid(5, 5)
    sim = _compile_cloth_sim(cloth_obj)
    assert sim.cloth_object is not None
    out_mesh = sim.cloth_object.object.data
    assert isinstance(out_mesh, bpy.types.Mesh)
    assert len(out_mesh.vertices) == 5 * 5


def test_cloth_faces_stored():
    """_cloth_faces should be a (M, 3) int array matching the triangulated mesh."""
    cloth_obj = _flat_grid(4, 4)
    sim = _compile_cloth_sim(cloth_obj)
    assert sim._cloth_faces is not None
    assert sim._cloth_faces.ndim == 2
    assert sim._cloth_faces.shape[1] == 3


# ---------------------------------------------------------------------------
# Step / dynamics tests
# ---------------------------------------------------------------------------


def test_cloth_falls_under_gravity():
    """Unhinged cloth should fall (centroid z decreases after several steps)."""
    cloth_obj = _flat_grid(4, 4)
    sim = _compile_cloth_sim(cloth_obj)
    assert sim.cloth_object is not None

    initial_z = float(sim.cloth_object.position[:, 2].mean())

    for _ in range(10):
        sim.step()

    positions = sim.cloth_object.position
    assert float(np.mean(positions[:, 2])) < initial_z


def test_cloth_pinned_vertices_do_not_fall():
    """Vertices with pinned=True (mass=0) should not move under gravity."""
    nx, ny = 4, 4
    cloth_obj = _flat_grid(nx, ny)
    mesh = cloth_obj.data
    assert isinstance(mesh, bpy.types.Mesh)

    # Pin the entire top row
    n = nx * ny
    pinned_data = np.zeros(n, dtype=bool)
    pinned_data[(ny - 1) * nx :] = True  # top row
    attr = mesh.attributes.new(name="pinned", type="BOOLEAN", domain="POINT")
    for i, val in enumerate(pinned_data):
        attr.data[i].value = bool(val)  # type: ignore[attr-defined]

    sim = _compile_cloth_sim(cloth_obj)
    assert sim.cloth_object is not None

    pinned_indices = np.where(pinned_data)[0]
    initial_z = sim.cloth_object.position[pinned_indices, 2].copy()

    for _ in range(10):
        sim.step()

    final_z = sim.cloth_object.position[pinned_indices, 2]
    np.testing.assert_allclose(final_z, initial_z, atol=1e-3)


def test_cloth_with_particles_correct_start_index():
    """When particles are added first, cloth particle_start should equal particle count."""
    cloth_obj = _flat_grid(3, 3)

    positions = np.zeros((5, 3), dtype=np.float32)
    positions[:, 2] = 10.0
    particle_bob = db.BlenderObject.from_pointcloud(positions, name="ParticleSource")

    man = get_manager(bpy.context)
    sim = SimulatorXPBD()
    man.add(sim)
    item = man.active_item
    item.particle_source = particle_bob.object
    item.cloth_source = cloth_obj
    item.device = "cpu"
    item.use_ground_plane = False
    item.substeps = 2
    sim.compile()

    assert sim._cloth_particle_start == 5
    assert sim._cloth_particle_count == 3 * 3
    assert sim.model.particle_count == 5 + 3 * 3
