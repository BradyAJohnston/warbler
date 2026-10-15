"""Demo scenes built with Geometry Nodes, checked like the Newton examples they copy."""

import bpy
import newton
import numpy as np
import pytest
from nodebpy import geometry as g

from warbler import demos
from warbler.manager import get_manager
from warbler.simulation import Simulator
from warbler.utils import get_scene


def _run(sim: Simulator, frames: int) -> np.ndarray:
    scene = get_scene()
    scene.frame_set(1)
    sim.compile()
    for f in range(2, frames + 2):
        scene.frame_set(f)
    assert sim.props.is_active, "simulation deactivated after an error"
    q = sim.particle_positions
    assert np.isfinite(q).all(), "particle positions contain NaN or inf"
    return q


def test_particle_grid():
    """add_particle_grid() from example_xpbd_vbd_coupled_solver, built in GN."""
    sim = demos.particles(bpy.context)
    source = sim.props.source
    assert source is not None
    # Newton's exact grid: 5 x 5 x 3 points
    for name, value in (("Dim X", 5), ("Dim Y", 5), ("Dim Z", 3)):
        demos.set_modifier_input(source, name, value)

    q0 = None
    sim.compile()
    q0 = sim.particle_positions.copy()
    assert sim.model.particle_count == 75
    assert len(sim.parts) == 1 and sim.parts[0].component == "points"
    np.testing.assert_allclose(sim.model.particle_mass.numpy(), 0.02, rtol=1e-6)  # ty: ignore[unresolved-attribute]
    np.testing.assert_allclose(sim.model.particle_radius.numpy(), 0.025, rtol=1e-6)  # ty: ignore[unresolved-attribute]
    # Grid starts at Newton's origin, spaced by cell size (+ jitter)
    np.testing.assert_allclose(q0.min(0), (-0.12, -0.12, 0.66), atol=0.003)
    np.testing.assert_allclose(q0.max(0), (0.08, 0.08, 0.76), atol=0.003)

    q = _run(sim, 120)

    # The example's test_final checks
    bbox = np.linalg.norm(q.max(0) - q.min(0))
    assert bbox < 10.0, f"bounding box exploded: {bbox:.2f}"
    assert q[:, 2].min() > -0.25, "excessive ground penetration"
    assert q[:, 2].max() < 3.0, "particles escaped upward"
    # ...and they actually fell to the ground
    assert q[:, 2].max() < 0.3

    out = sim.parts[0].output
    assert out is not None
    np.testing.assert_allclose(out.position, q, atol=1e-6)


def test_hanging_cloth():
    """example_cloth_hanging (VBD) with the grid built and pinned in GN."""
    sim = demos.cloth(bpy.context)
    sim.compile()

    width, height = 64, 32
    assert sim.model.particle_count == (width + 1) * (height + 1)
    assert sim.model.tri_count == width * height * 2
    assert sim.parts[0].is_cloth
    assert sim._pinned_indices is not None
    pinned = sim._pinned_indices
    assert len(pinned) == height + 1
    q0 = sim.particle_positions.copy()
    # fix_left pins the grid's x = 0 column, which lies along world X after
    # Newton's 90° rotation about Z
    np.testing.assert_allclose(q0[pinned, 1], 0.0, atol=1e-5)
    np.testing.assert_allclose(q0[pinned, 2], 4.0, atol=1e-5)

    q = _run(sim, 60)

    # The example's test_final checks
    assert (q[:, 2] > 0.0).all(), "particles are above the ground"
    lower = np.array([-width * 0.11, -4.0, -1.8])
    upper = np.array([0.1, 7.0, 4.0])
    assert ((q >= lower) & (q <= upper)).all(), "particles within a reasonable volume"
    # Pins hold, the free edge swings down
    np.testing.assert_allclose(q[pinned], q0[pinned], atol=1e-5)
    assert q[:, 2].min() < 2.0


def test_particles_report_collisions():
    """Settled particles touching the ground are flagged ``is_colliding``."""
    sim = demos.particles(bpy.context)
    source = sim.props.source
    assert source is not None
    for name, value in (("Dim X", 5), ("Dim Y", 5), ("Dim Z", 3)):
        demos.set_modifier_input(source, name, value)
    q = _run(sim, 120)
    out = sim.parts[0].output
    assert out is not None
    colliding = out.named_attribute("is_colliding")
    z = q[:, 2]
    assert colliding[z < z.min() + 0.01].all()
    assert not colliding[z > 0.06].any()
    # XPBD doesn't report particle contact forces
    assert "contact_force" not in out.object.data.attributes  # ty: ignore[unresolved-attribute]


def test_carried_cloth_attachment_and_contact_forces():
    """``attach`` holds the cloth edge to an animated bar (VBD); the cloth
    lands on the obstacle, which reports the contact force."""
    sim = demos.carried_cloth(bpy.context)
    get_scene().frame_set(1)
    sim.compile()
    edge = sim.particle_positions[:, 1] < 0.01
    assert sim.model.attachment_particle_body_count == edge.sum() == 33
    q0 = sim.particle_positions.copy()

    q = _run(sim, 80)
    bar = bpy.data.objects["Bar"]
    moved = np.array(bar.matrix_world.translation) - (-1.6, -0.15, 3.0)
    np.testing.assert_allclose((q[edge] - q0[edge]).mean(0), moved, atol=0.05)

    out = sim.parts[0].output
    assert out is not None
    colliding = out.named_attribute("is_colliding")
    force = out.named_attribute("contact_force")
    assert colliding.sum() > 50
    assert force[colliding][:, 2].sum() > 0
    assert not force[~colliding].any()
    # The obstacle carries part of the cloth's weight
    weight = 9.81 * sim.model.particle_mass.numpy().sum()  # ty: ignore[unresolved-attribute]
    fz = bpy.data.objects["Obstacle"].wb.contact_force[2]  # ty: ignore[unresolved-attribute]
    assert -weight < fz < -0.2 * weight


@pytest.mark.parametrize(
    ("log_barrier", "dat"), [(True, False), (False, True)], ids=["log_barrier", "dat"]
)
def test_vbd_rigid_soft_contact_options(log_barrier, dat):
    """Newton 1.7's opt-in VBD rigid-soft contact modes run (DAT makes the
    solver own the collision pipeline)."""
    sim = demos.carried_cloth(bpy.context)
    sim.props.rigid_soft_log_barrier = log_barrier
    sim.props.rigid_soft_dat = dat
    q = _run(sim, 60)
    assert sim._solver_collides == dat
    assert q[:, 2].min() > 0.5, "cloth stays on the obstacle"
    out = sim.parts[0].output
    assert out is not None
    assert out.named_attribute("is_colliding").sum() > 50


def test_hanging_ropes_follow_animated_pins():
    """Pinned curve roots are re-read from GN each frame and drag the ropes."""
    sim = demos.ropes(bpy.context)
    q = _run(sim, 60)

    part = sim.parts[0]
    assert part.component == "curves"
    assert part.curve_sizes is not None
    assert list(part.curve_sizes) == [32] * 8
    assert sim._pinned_indices is not None
    assert len(sim._pinned_indices) == 8

    # Pinned roots match the GN-animated positions on the current frame
    source = sim.props.source
    assert source is not None
    from warbler.geometryset import GeometrySet

    gn = GeometrySet(source).curves.to_props()
    pinned = sim._pinned_indices
    np.testing.assert_allclose(q[pinned], gn["position"][pinned], atol=1e-5)

    # Free ends have swung below their roots and the springs haven't torn
    ends = pinned + 31
    assert (q[ends, 2] < q[pinned, 2] - 1.0).all()
    seg = np.linalg.norm(np.diff(q.reshape(8, 32, 3), axis=1), axis=2)
    rest = 2.0 / 31
    assert seg.max() < rest * 2.0


def test_combined_source_splits_into_parts():
    """Points, curves and a mesh from one GN tree become three output objects."""
    with g.tree("Mixed", clear=True, is_modifier=True) as tree:
        points = g.Points(
            10, position=g.RandomValue.vector(min=(-1, -1, 1), max=(1, 1, 2))
        )
        curve = g.CurveLine(start=(0, 0, 3), end=(1, 0, 3)) >> g.ResampleCurve(count=5)
        cloth = g.TransformGeometry(g.Grid(1, 1, 4, 4), translation=(0, 0, 4))
        g.JoinGeometry([points, curve, cloth]) >> tree.outputs.geometry("Geometry")

    obj = demos._source_object("Mixed", tree.tree)
    man = get_manager(bpy.context)
    sim = Simulator()
    man.add(sim)
    sim.props.source = obj
    sim.props.device = "cpu"
    sim.props.solver = "VBD"  # XPBD would also add springs along the cloth edges
    sim.compile()

    assert [(p.component, p.start, p.count) for p in sim.parts] == [
        ("points", 0, 10),
        ("curves", 10, 5),
        ("mesh", 15, 16),
    ]
    assert sim.parts[2].is_cloth
    assert sim.model.spring_count == 4 + 3  # curve stretch + bend
    for _ in range(10):
        sim.step()
    assert np.isfinite(sim.particle_positions).all()
    outputs = [p.output.object for p in sim.parts if p.output is not None]
    assert [o.type for o in outputs] == ["POINTCLOUD", "CURVES", "MESH"]

    # Recompiling replaces, rather than duplicates, the output objects
    names = {o.name for o in outputs}
    sim.compile()
    assert {p.output.name for p in sim.parts if p.output is not None} == names
    assert sum(o.name.startswith("Mixed_") for o in bpy.data.objects) == 3


@pytest.mark.parametrize("demo", list(demos.DEMOS))
def test_add_demo_operator(demo):
    assert bpy.ops.wb.add_demo(demo=demo) == {"FINISHED"}  # ty: ignore[unresolved-attribute]
    item = get_manager(bpy.context).active_item
    assert item.is_compiled
    assert item.source is not None


def test_recompile_reuses_output_objects():
    """Re-compiling rebuilds output geometry in place, keeping modifiers."""
    sim = demos.particles(bpy.context)
    source = sim.props.source
    assert source is not None
    demos.set_modifier_input(source, "Dim Z", 2)
    sim.compile()
    out = sim.parts[0].output
    assert out is not None
    obj, data = out.object, out.object.data
    obj.modifiers.new("Keep Me", "NODES")
    assert len(out) == 16 * 16 * 2

    # Topology change: more points from the same source
    demos.set_modifier_input(source, "Dim Z", 3)
    sim.compile()
    out = sim.parts[0].output
    assert out is not None
    assert out.object == obj and out.object.data == data
    assert "Keep Me" in obj.modifiers
    assert len(out) == 16 * 16 * 3
    np.testing.assert_allclose(out.position, sim.particle_positions, atol=1e-6)

    # A fresh simulation for the same source finds the output by name
    sim2 = demos.particles(bpy.context)
    sim2.compile()
    assert sim2.parts[0].output is not None
    assert sim2.parts[0].output.object == obj


def test_recompile_rebuilds_cloth_and_curves_in_place():
    for build in (demos.cloth, demos.ropes):
        sim = build(bpy.context)
        sim.compile()
        out = sim.parts[0].output
        assert out is not None
        obj, data = out.object, out.object.data
        sim.step()
        sim.compile()
        out = sim.parts[0].output
        assert out is not None
        assert out.object == obj and out.object.data == data
        # Geometry was reset to the compiled start state
        np.testing.assert_allclose(out.position, sim.particle_positions, atol=1e-6)


def test_free_simulation():
    """Free stops stepping, keeps outputs, and allows changing the solver."""
    sim = demos.ropes(bpy.context)
    assert bpy.ops.wb.compile_simulation() == {"FINISHED"}  # ty: ignore[unresolved-attribute]
    out = sim.parts[0].output
    assert out is not None
    obj = out.object

    assert bpy.ops.wb.free_simulation() == {"FINISHED"}  # ty: ignore[unresolved-attribute]
    assert not sim.props.is_compiled
    assert not hasattr(sim, "model")
    assert obj.name in bpy.data.objects

    # Frame changes no longer step it
    before = out.position.copy()
    get_scene().frame_set(get_scene().frame_current + 5)
    np.testing.assert_array_equal(out.position, before)

    sim.props.solver = "VBD"
    sim.props.iterations = 5
    sim.compile()
    assert isinstance(sim.solver, newton.solvers.SolverVBD)
    assert sim.parts[0].output is not None
    assert sim.parts[0].output.object == obj


def test_free_without_live_simulation_unlocks_settings():
    """After a file reload the item says compiled but has no simulation object."""
    man = get_manager(bpy.context)
    man.add(Simulator())
    item = man.active_item
    item.is_compiled = True
    man.simulations.clear()
    assert bpy.ops.wb.free_simulation() == {"FINISHED"}  # ty: ignore[unresolved-attribute]
    assert not item.is_compiled
