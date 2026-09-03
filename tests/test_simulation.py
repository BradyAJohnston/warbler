import bpy
import pytest

from warbler.manager import get_manager
from warbler.props import object_properties
from warbler.simulation import SimulatorXPBD
from warbler.utils import get_scene


def _make_rigid_collection(name: str = "Rigid") -> bpy.types.Collection:
    coll = bpy.data.collections.new(name)
    get_scene().collection.children.link(coll)
    return coll


def test_empty_simulation_compiles():
    """A simulation with no objects and no particle source should compile cleanly."""
    man = get_manager(bpy.context)
    man.add(SimulatorXPBD())
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
    man.add(SimulatorXPBD())
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
