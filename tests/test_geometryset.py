import bpy
import databpy as db
import numpy as np

from warbler.geometryset import GeometrySet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _random_positions(n: int) -> np.ndarray:
    return RNG.random((n, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# Mesh tests
# ---------------------------------------------------------------------------


def test_reads_positions_from_mesh():
    """GeometrySet should expose vertex positions from a plain mesh object."""
    cube = bpy.data.objects["Cube"]
    geo = GeometrySet(cube)
    props = geo.pointcloud.to_props()

    assert "position" in props
    positions = props["position"]
    assert isinstance(positions, np.ndarray)
    assert positions.ndim == 2
    assert positions.shape[1] == 3
    assert len(positions) == len(cube.data.vertices)


def test_obj_type_is_mesh():
    cube = bpy.data.objects["Cube"]
    geo = GeometrySet(cube)
    assert geo.obj_type == "MESH"


def test_to_props_skips_missing_attributes():
    """to_props() should only include attributes that exist; no KeyError for missing ones."""
    cube = bpy.data.objects["Cube"]
    geo = GeometrySet(cube)
    props = geo.pointcloud.to_props()

    assert "velocity" not in props
    assert "mass" not in props
    assert "radius" not in props


def test_reads_custom_named_attribute_on_mesh():
    """A named float attribute added to a mesh should appear in to_props()."""
    cube = bpy.data.objects["Cube"]
    mesh = cube.data

    attr = mesh.attributes.new(name="mass", type="FLOAT", domain="POINT")
    for i, d in enumerate(attr.data):
        d.value = float(i + 1)

    geo = GeometrySet(cube)
    props = geo.pointcloud.to_props()

    assert "mass" in props
    mass = props["mass"]
    assert isinstance(mass, np.ndarray)
    assert len(mass) == len(mesh.vertices)


def test_mesh_point_count():
    cube = bpy.data.objects["Cube"]
    geo = GeometrySet(cube)
    assert geo._get_point_count() == len(cube.data.vertices)


# ---------------------------------------------------------------------------
# Point cloud tests
# ---------------------------------------------------------------------------


def test_reads_positions_from_pointcloud():
    """GeometrySet reads positions from a databpy point cloud object."""
    positions = _random_positions(50)
    bob = db.BlenderObject.from_pointcloud(positions, name="TestPC")

    geo = GeometrySet(bob.object)
    props = geo.pointcloud.to_props()

    assert "position" in props
    assert props["position"].shape == (50, 3)
    np.testing.assert_allclose(props["position"], positions, atol=1e-5)


def test_pointcloud_point_count():
    positions = _random_positions(30)
    bob = db.BlenderObject.from_pointcloud(positions, name="TestPC_count")

    geo = GeometrySet(bob.object)
    assert geo._get_point_count() == 30


def test_pointcloud_reads_radius_attribute():
    """radius stored on a point cloud should be picked up by to_props()."""
    n = 20
    positions = _random_positions(n)
    bob = db.BlenderObject.from_pointcloud(positions, name="TestPC_radius")
    radii = np.full(n, 0.25, dtype=np.float32)
    bob.store_named_attribute(radii, "radius")

    geo = GeometrySet(bob.object)
    props = geo.pointcloud.to_props()

    assert "radius" in props
    np.testing.assert_allclose(props["radius"], radii, atol=1e-5)


def test_pointcloud_reads_velocity_and_mass():
    """velocity and mass attributes on a point cloud should both appear in to_props()."""
    n = 10
    positions = _random_positions(n)
    bob = db.BlenderObject.from_pointcloud(positions, name="TestPC_velm")

    velocities = RNG.random((n, 3), dtype=np.float32)
    masses = np.ones(n, dtype=np.float32) * 2.5

    bob.store_named_attribute(velocities, "velocity")
    bob.store_named_attribute(masses, "mass")

    geo = GeometrySet(bob.object)
    props = geo.pointcloud.to_props()

    assert set(props.keys()) == {"position", "velocity", "mass"}
    np.testing.assert_allclose(props["velocity"], velocities, atol=1e-5)
    np.testing.assert_allclose(props["mass"], masses, atol=1e-5)


def test_pointcloud_to_props_returns_all_four_particle_attrs():
    """When all four known attributes are present they all appear in to_props()."""
    n = 15
    positions = _random_positions(n)
    bob = db.BlenderObject.from_pointcloud(positions, name="TestPC_all")

    bob.store_named_attribute(np.zeros((n, 3), dtype=np.float32), "velocity")
    bob.store_named_attribute(np.ones(n, dtype=np.float32), "mass")
    bob.store_named_attribute(np.full(n, 0.1, dtype=np.float32), "radius")

    geo = GeometrySet(bob.object)
    props = geo.pointcloud.to_props()

    assert set(props.keys()) == {"position", "velocity", "mass", "radius"}
    assert all(isinstance(v, np.ndarray) for v in props.values())


# ---------------------------------------------------------------------------
# Vertex-only mesh (no faces) tests
# ---------------------------------------------------------------------------


def test_reads_positions_from_vertex_mesh():
    """A mesh with only vertices (no faces) should work as a particle source."""
    positions = _random_positions(25)
    bob = db.BlenderObject.from_mesh(vertices=positions, name="VertexMesh")

    geo = GeometrySet(bob.object)
    props = geo.pointcloud.to_props()

    assert "position" in props
    assert props["position"].shape == (25, 3)


def test_vertex_mesh_point_count():
    positions = _random_positions(40)
    bob = db.BlenderObject.from_mesh(vertices=positions, name="VertexMesh_count")

    geo = GeometrySet(bob.object)
    assert geo._get_point_count() == 40
