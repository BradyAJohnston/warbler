import bpy
import numpy as np

from warbler.geometryset import GeometrySet


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

    # Default cube has no velocity/mass/radius attributes
    assert "velocity" not in props
    assert "mass" not in props
    assert "radius" not in props


def test_reads_custom_named_attribute():
    """A named float attribute added to a mesh should appear in the attributes collection."""
    cube = bpy.data.objects["Cube"]
    mesh = cube.data

    # Add a custom 'mass' attribute on POINT domain
    attr = mesh.attributes.new(name="mass", type="FLOAT", domain="POINT")
    for i, d in enumerate(attr.data):
        d.value = float(i + 1)

    geo = GeometrySet(cube)
    props = geo.pointcloud.to_props()

    assert "mass" in props
    mass = props["mass"]
    assert isinstance(mass, np.ndarray)
    assert len(mass) == len(mesh.vertices)


def test_point_count():
    cube = bpy.data.objects["Cube"]
    geo = GeometrySet(cube)
    assert geo._get_point_count() == len(cube.data.vertices)
