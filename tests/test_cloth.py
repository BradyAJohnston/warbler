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
    cloth_obj: bpy.types.Object, device: str = "cpu", substeps: int = 10
) -> SimulatorXPBD:
    man = get_manager(bpy.context)
    sim = SimulatorXPBD()
    man.add(sim)
    item = man.active_item
    item.cloth_source = cloth_obj
    item.device = device
    item.use_ground_plane = True
    item.substeps = substeps
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
    """After compilation cloth_object wraps the source mesh; no separate WarblerCloth."""
    cloth_obj = _flat_grid(4, 4)
    sim = _compile_cloth_sim(cloth_obj)
    assert sim.cloth_object is not None
    assert sim.cloth_object.object is cloth_obj
    assert bpy.data.objects.get("WarblerCloth") is None


def test_cloth_output_vertex_count_matches_source():
    cloth_obj = _flat_grid(5, 5)
    sim = _compile_cloth_sim(cloth_obj)
    assert sim.cloth_object is not None
    out_mesh = sim.cloth_object.object.data
    assert isinstance(out_mesh, bpy.types.Mesh)
    assert len(out_mesh.vertices) == 5 * 5


def test_cloth_position_start_attribute():
    """Source mesh should gain position_start storing the initial local positions."""
    cloth_obj = _flat_grid(4, 4)
    mesh = cloth_obj.data
    assert isinstance(mesh, bpy.types.Mesh)
    original = np.array([v.co for v in mesh.vertices], dtype=np.float32)

    _compile_cloth_sim(cloth_obj)

    assert "position_start" in mesh.attributes
    stored = np.array(
        [d.vector for d in mesh.attributes["position_start"].data],  # type: ignore[attr-defined]
        dtype=np.float32,
    )
    np.testing.assert_allclose(stored, original, atol=1e-5)


def test_cloth_default_plane_stable():
    """A coarse 2x2m plane at z=1 should fall under gravity without exploding."""
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 1.0))
    plane = bpy.context.active_object
    assert plane is not None

    sim = _compile_cloth_sim(plane)

    for _ in range(5):
        sim.step()

    assert sim.state_0.particle_q is not None
    positions = sim.state_0.particle_q.numpy()[: sim._cloth_particle_count]
    assert np.all(np.isfinite(positions)), "Cloth positions are NaN/inf"
    mean_z = float(positions[:, 2].mean())
    assert mean_z < 0.95, (
        f"Cloth barely moved (z={mean_z:.4f}); particle_radius too large?"
    )


def test_cloth_cube_stable():
    """A default Blender cube used as cloth should not explode after several frames.

    spring_ke=1e3 requires substeps >= 8 at 30 fps (ke*dt^2 < ~0.025).
    The default substeps=10 satisfies this.  Regression: earlier defaults caused
    positions to reach ~3000 by frame 3.
    """
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 2.0))
    cube = bpy.context.active_object
    assert cube is not None

    sim = _compile_cloth_sim(cube)

    assert sim._cloth_particle_count == 8
    assert sim._cloth_faces is not None

    for i in range(5):
        sim.step()
        assert sim.state_0.particle_q is not None
        positions = sim.state_0.particle_q.numpy()[: sim._cloth_particle_count]
        assert np.all(np.isfinite(positions)), f"Explosion at step {i + 1}"
        assert np.max(np.abs(positions)) < 100.0, (
            f"Positions unreasonably large at step {i + 1}: {np.max(np.abs(positions)):.2e}"
        )


def test_cloth_grid_ground_impact_stable():
    """A fine grid dropped onto the ground must not explode on impact.

    Regression: density-derived per-vertex mass (~1e-4 kg) made the XPBD
    penalty-contact solve diverge to NaN on the first ground contact (~frame
    15), independent of substeps or spring_ke. The cloth_mass_min floor keeps
    contacts stable. Run past the impact frame at low substeps to catch it.
    """
    grid = _flat_grid(12, 12, spacing=0.1)
    sim = _compile_cloth_sim(grid, substeps=2)

    n = sim._cloth_particle_count
    for i in range(40):
        sim.step()
        assert sim.state_0.particle_q is not None
        positions = sim.state_0.particle_q.numpy()[:n]
        assert np.all(np.isfinite(positions)), f"Explosion (NaN) at step {i + 1}"
        assert np.max(np.abs(positions)) < 50.0, (
            f"Positions diverged at step {i + 1}: {np.max(np.abs(positions)):.2e}"
        )

    # Cloth should have come to rest on the ground, not passed through or blown up.
    assert sim.state_0.particle_q is not None
    final_z = float(sim.state_0.particle_q.numpy()[:n, 2].mean())
    assert -0.5 < final_z < 0.5, f"Cloth not resting near ground (z={final_z:.3f})"


def test_cloth_subdivided_cube_stable():
    """A subdivided cube must not explode or gain energy (launch upward).

    Regression: XPBD springs are only conditionally stable -- Newton's solver
    sums every spring delta at a shared vertex with no constraint averaging, so
    a denser mesh (lower per-vertex mass, more shared springs) diverged where a
    plain 8-vertex cube was fine. The solver now auto-subdivides the frame into
    enough substeps to keep spring_ke*dt^2/mass in the stable regime.
    """
    import bmesh

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 3.0))
    cube = bpy.context.active_object
    assert cube is not None and isinstance(cube.data, bpy.types.Mesh)

    bm = bmesh.new()
    bm.from_mesh(cube.data)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=2, use_grid_fill=True)
    bm.to_mesh(cube.data)
    bm.free()

    sim = _compile_cloth_sim(cube, substeps=10)
    # Stability criterion should have raised the effective substep count.
    assert sim._stable_substeps() > 10

    n = sim._cloth_particle_count
    assert sim.state_0.particle_q is not None
    z_start = float(sim.state_0.particle_q.numpy()[:n, 2].mean())

    max_z = z_start
    for i in range(60):
        sim.step()
        assert sim.state_0.particle_q is not None
        p = sim.state_0.particle_q.numpy()[:n]
        assert np.all(np.isfinite(p)), f"Explosion (NaN) at step {i + 1}"
        max_z = max(max_z, float(p[:, 2].mean()))

    # Cloth falls onto the ground; it must never gain energy and launch upward.
    assert max_z < z_start + 1.0, (
        f"Cloth gained energy and launched (z_start={z_start:.2f}, max_z={max_z:.2f})"
    )


def test_cloth_writes_back_to_source():
    """Stepping the simulation should update vertex positions on the source mesh."""
    cloth_obj = _flat_grid(4, 4)
    sim = _compile_cloth_sim(cloth_obj)
    assert sim.cloth_object is not None

    initial_z = float(sim.cloth_object.position[:, 2].mean())

    for _ in range(10):
        sim.step()

    final_z = float(sim.cloth_object.position[:, 2].mean())
    assert final_z < initial_z, (
        "Source mesh vertices did not move after simulation steps"
    )


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
