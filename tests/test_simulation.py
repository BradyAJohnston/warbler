import bpy
import pytest

from warbler.manager import get_manager
from warbler.props import object_properties
from warbler.simulation import Simulator
from warbler.utils import get_scene


def _make_rigid_collection(name: str = "Rigid") -> bpy.types.Collection:
    coll = bpy.data.collections.new(name)
    get_scene().collection.children.link(coll)
    return coll


def test_empty_simulation_compiles():
    """A simulation with no objects and no particle source should compile cleanly."""
    man = get_manager(bpy.context)
    man.add(Simulator())
    item = man.active_item
    item.device = "cpu"
    item.use_ground_plane = False

    man.active_simulation.compile()
    assert item.is_compiled


def test_rigid_body_compiles():
    """A simulation with one CUBE rigid body should compile without error."""
    cube = bpy.data.objects["Cube"]
    object_properties(cube).sim_shape = "CUBE"

    coll = _make_rigid_collection()
    coll.objects.link(cube)

    man = get_manager(bpy.context)
    man.add(Simulator())
    item = man.active_item
    item.sim_rigid_collection = coll
    item.device = "cpu"

    man.active_simulation.compile()
    assert item.is_compiled


def test_active_body_falls(rigid_simulation):
    """An active rigid body should move downward after several simulation steps."""
    cube = bpy.data.objects["Cube"]
    initial_z = float(cube.location.z)

    for _ in range(20):
        rigid_simulation.step()

    assert cube.location.z < initial_z, (
        f"Expected body to fall below z={initial_z:.3f}, got z={cube.location.z:.3f}"
    )


def test_inactive_body_does_not_move_from_physics(rigid_simulation):
    """An inactive (kinematic) body should stay where Blender puts it."""
    cube = bpy.data.objects["Cube"]
    object_properties(cube).is_active = False

    initial_z = float(cube.location.z)

    for _ in range(20):
        rigid_simulation.step()

    assert cube.location.z == pytest.approx(initial_z, abs=1e-4), (
        "Inactive body should not move from physics; "
        f"moved from z={initial_z:.4f} to z={cube.location.z:.4f}"
    )


def test_step_increments_clock(rigid_simulation):
    assert rigid_simulation.clock == 0
    rigid_simulation.step()
    assert rigid_simulation.clock == 1
    rigid_simulation.step()
    assert rigid_simulation.clock == 2


def test_body_q_is_populated_after_compile(rigid_simulation):
    """state_0.body_q must be a numpy array with one row after compilation."""
    body_q = rigid_simulation.state_0.body_q
    assert body_q is not None
    arr = body_q.numpy()
    assert arr.shape == (1, 7), f"Expected shape (1, 7), got {arr.shape}"


def test_simulation_timing_recorded(rigid_simulation):
    """time_compute and time_sync should be non-zero after a step."""
    rigid_simulation.step()
    item = rigid_simulation.props
    assert item.time_compute > 0
    assert item.time_sync > 0


def test_rigid_box_scales_live():
    """Scaling a rigid box in Blender resizes its collision shape without re-compiling."""
    import databpy as db
    import numpy as np

    cube = bpy.data.objects["Cube"]
    cube.location = (0, 0, 1)  # 2 m cube, top at z = 2
    object_properties(cube).sim_shape = "CUBE"
    coll = _make_rigid_collection()
    coll.objects.link(cube)

    grid = np.mgrid[-0.4:0.41:0.2, -0.4:0.41:0.2].reshape(2, -1).T
    positions = np.column_stack([grid, np.full(len(grid), 2.5)])
    source = db.BlenderObject.from_pointcloud(positions, name="Drops")
    source.store_named_attribute(np.full(len(grid), 0.05, dtype=np.float32), "radius")

    man = get_manager(bpy.context)
    sim = Simulator()
    man.add(sim)
    item = man.active_item
    item.sim_rigid_collection = coll
    item.source = source.object
    item.device = "cpu"
    sim.compile()

    for _ in range(48):
        sim.step()
    z_rest = sim.particle_positions[:, 2]
    assert np.allclose(z_rest, 2.05, atol=0.05), z_rest

    # Grow the box over one second (24 frames) until its top is at z = 2.5;
    # the particles should be carried up, not launched
    for frame in range(1, 49):
        cube.scale = (1, 1, 1 + 0.5 * min(frame / 24, 1.0))
        bpy.context.view_layer.update()  # ty: ignore[unresolved-attribute]
        sim.step()
        assert sim.particle_positions[:, 2].max() < 3.0

    shape = object_properties(cube).sim_body_index
    np.testing.assert_allclose(sim.model.shape_scale.numpy()[shape], (1, 1, 1.5))  # ty: ignore[unresolved-attribute]
    np.testing.assert_allclose(
        sim.model.shape_collision_radius.numpy()[shape],  # ty: ignore[unresolved-attribute]
        np.linalg.norm((1, 1, 1.5)),
        rtol=1e-6,
    )
    z = sim.particle_positions[:, 2]
    assert np.allclose(z, 2.55, atol=0.05), z


def _parented_cube(parent_location, parent_rotation_z, child_location):
    import math

    parent = bpy.data.objects.new("Parent", None)
    get_scene().collection.objects.link(parent)
    parent.location = parent_location
    parent.rotation_euler = (0, 0, math.radians(parent_rotation_z))
    cube = bpy.data.objects["Cube"]
    cube.parent = parent
    cube.location = child_location
    object_properties(cube).sim_shape = "CUBE"
    coll = _make_rigid_collection()
    coll.objects.link(cube)

    man = get_manager(bpy.context)
    sim = Simulator()
    man.add(sim)
    sim.props.sim_rigid_collection = coll
    sim.props.device = "cpu"
    sim.props.use_ground_plane = False
    return sim, parent, cube


def test_kinematic_body_uses_world_transform():
    """A parented collider is placed at its world-space transform."""
    import numpy as np

    sim, parent, _cube = _parented_cube((2, 0, 3), 90, (1, 0, 0))
    sim.compile()
    body = sim.state_0.body_q.numpy()[0]
    np.testing.assert_allclose(body[:3], (2, 1, 3), atol=1e-5)
    # 90° about Z, xyzw
    np.testing.assert_allclose(
        np.abs(body[3:]), (0, 0, 0.70710677, 0.70710677), atol=1e-5
    )

    # Moving the parent moves the collider
    parent.location = (2, 0, 4)
    bpy.context.view_layer.update()  # ty: ignore[unresolved-attribute]
    sim.step()
    sim.step()
    body = sim.state_0.body_q.numpy()[0]
    np.testing.assert_allclose(body[:3], (2, 1, 4), atol=1e-4)


def test_spinning_kinematic_body_has_angular_velocity():
    import math

    import numpy as np

    sim, parent, _cube = _parented_cube((0, 0, 0), 0, (0, 0, 0))
    sim.compile()
    sim.step()
    step = math.radians(5)
    parent.rotation_euler = (0, 0, step)
    bpy.context.view_layer.update()  # ty: ignore[unresolved-attribute]
    sim._update_simulation_from_blender()
    qd = sim.state_0.body_qd.numpy()[0]
    np.testing.assert_allclose(qd[3:], (0, 0, step * sim.fps), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(qd[:3], 0, atol=1e-6)


def test_active_body_writes_back_through_parent():
    """An active body under a transformed parent ends up at the sim's world transform."""
    import numpy as np

    sim, parent, cube = _parented_cube((0, 0, 3), 45, (0, 0, 2))
    object_properties(cube).is_active = True
    sim.props.use_ground_plane = True
    sim.compile()
    for _ in range(10):
        sim.step()
    body = sim.state_0.body_q.numpy()[0]
    world = np.array(cube.matrix_world.translation)
    np.testing.assert_allclose(world, body[:3], atol=1e-4)
    assert world[2] < 5.0
    assert cube.parent == parent
