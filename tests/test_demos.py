"""Demo scenes built with Geometry Nodes, checked like the Newton examples they copy."""

import bpy
import databpy as db
import newton
import numpy as np
import pytest
import warp as wp
from nodebpy import geometry as g
from nodebpy.builder.asset import build_from_source
from scipy.spatial.transform import Rotation

from warbler import demos, nodes
from warbler.geometryset import GeometrySet
from warbler.manager import get_manager
from warbler.simulation import COLLIDERS, Simulator
from warbler.utils import get_scene

# Set Collider's menu, in COLLIDERS order
SHAPES = ["Convex Hull", "Box", "Sphere", "Capsule", "Mesh"]


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
    solver own the collision pipeline), recorded as a CUDA graph."""
    sim = demos.carried_cloth(bpy.context)
    sim.props.rigid_soft_log_barrier = log_barrier
    sim.props.rigid_soft_dat = dat
    q = _run(sim, 60)
    assert sim._solver_collides == dat
    assert sim._graph is not None or not wp.get_cuda_device_count()
    assert q[:, 2].min() > 0.5, "cloth stays on the obstacle"
    out = sim.parts[0].output
    assert out is not None
    assert out.named_attribute("is_colliding").sum() > 50


@pytest.mark.parametrize("substeps", [10, 9], ids=["even", "odd"])
def test_cuda_graph_matches_plain_launches(substeps):
    """A recorded frame replays exactly what launching kernel by kernel does."""
    if not wp.get_cuda_device_count():
        pytest.skip("needs CUDA")
    results = []
    for use_graph in (False, True):
        bpy.ops.wm.read_homefile(use_empty=True)
        sim = demos.cloth(bpy.context)
        sim.props.substeps = substeps
        sim.props.use_cuda_graph = use_graph
        results.append(_run(sim, 20))
        assert (sim._graph is not None) == use_graph
    np.testing.assert_array_equal(results[0], results[1])


def test_cuda_graph_rerecords_on_setting_change():
    if not wp.get_cuda_device_count():
        pytest.skip("needs CUDA")
    sim = demos.particles(bpy.context)
    sim.props.drag = 1.0
    sim.props.temperature = 1e-4
    _run(sim, 5)
    graph = sim._graph
    assert graph is not None
    # The thermostat seed advances on the device, so replays draw fresh noise
    assert sim._noise_seed.numpy()[0] == 5 * sim.substeps
    sim.props.substeps = 4
    scene = get_scene()
    scene.frame_set(scene.frame_current + 1)
    assert sim._graph is not None and sim._graph is not graph
    assert sim._graph_key is not None and sim._graph_key[0] == 4


@pytest.mark.parametrize("self_contact", [False, True], ids=["off", "on"])
def test_cloth_self_collision(self_contact):
    """A cloth piled up by lowering its bar keeps its layers apart only with
    VBD self-contact."""
    from scipy.spatial import KDTree

    sim = demos.carried_cloth(bpy.context)
    sim.props.cloth_self_contact = self_contact
    bpy.data.objects.remove(bpy.data.objects["Obstacle"])
    bar = bpy.data.objects["Bar"]
    bar.animation_data_clear()
    bar.location = (-1.6, -0.15, 3.0)
    bar.keyframe_insert("location", frame=30)
    bar.location = (-1.6, -0.15, 0.3)
    bar.keyframe_insert("location", frame=100)
    q = _run(sim, 150)

    # Closest approach of vertices that are far apart on the cloth itself
    rest = sim.model.particle_q.numpy()  # ty: ignore[unresolved-attribute]
    pairs = KDTree(q).query_pairs(0.1, output_type="ndarray")
    far = np.linalg.norm(rest[pairs[:, 0]] - rest[pairs[:, 1]], axis=1) > 0.25
    separation = np.linalg.norm(q[pairs[far, 0]] - q[pairs[far, 1]], axis=1)
    assert len(separation) > 50, "the cloth folds onto itself"
    margin = sim.props.cloth_self_contact_margin
    if self_contact:
        assert separation.min() > 0.8 * margin
    else:
        assert separation.min() < 0.5 * margin


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


def _small_monkeys(dims=(4, 4, 2)) -> Simulator:
    sim = demos.falling_monkeys(bpy.context)
    source = sim.props.source
    assert source is not None
    for name, value in zip(("Dim X", "Dim Y", "Dim Z"), dims, strict=True):
        demos.set_modifier_input(source, name, value)
    # Room for a turned monkey's bounding box: overlapping boxes start in
    # deep contact and are thrown apart
    demos.set_modifier_input(source, "Spacing", 2.0)
    return sim


def _run_bodies(sim: Simulator, frames: int) -> np.ndarray:
    scene = get_scene()
    scene.frame_set(1)
    sim.compile()
    for f in range(2, frames + 2):
        scene.frame_set(f)
    assert sim.props.is_active, "simulation deactivated after an error"
    q = sim.state_0.body_q.numpy()  # ty: ignore[unresolved-attribute]
    assert np.isfinite(q).all()
    return q


@pytest.mark.parametrize("collider", range(5), ids=list(COLLIDERS))
@pytest.mark.parametrize("solver", ["XPBD", "VBD"])
def test_falling_monkeys_settle(solver, collider):
    """Instances become rigid bodies with the INSTANCE ``collider`` shape, fall
    and come to rest on the ground."""
    sim = _small_monkeys()
    sim.props.solver = solver
    source = sim.props.source
    assert source is not None
    demos.set_modifier_input(source, "Collider", SHAPES[collider])
    q = _run_bodies(sim, 300)
    assert sim.model.body_count == 32
    assert (
        int(sim.model.shape_type.numpy()[1])  # ty: ignore[unresolved-attribute]
        == {
            0: newton.GeoType.CONVEX_MESH,
            1: newton.GeoType.BOX,
            2: newton.GeoType.SPHERE,
            3: newton.GeoType.CAPSULE,
            4: newton.GeoType.MESH,
        }[collider]
    )
    assert q[:, 2].min() > 0.05, "fell through the ground"
    assert q[:, 2].max() < 3.0, "still in the air"
    if COLLIDERS[collider] in ("SPHERE", "CAPSULE"):
        return  # round shapes roll on: there's no rolling resistance
    speed = np.linalg.norm(sim.state_0.body_qd.numpy()[:, :3], axis=1)  # ty: ignore[unresolved-attribute]
    assert np.median(speed) < 0.1


def test_instances_output_follows_bodies():
    """The output re-places the source's instances at the simulated transforms,
    keeping their scale."""
    sim = _small_monkeys()
    q = _run_bodies(sim, 30)
    assert sim.instances is not None and sim.instances.output is not None
    obj = sim.instances.output.object
    geo = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).evaluated_geometry()
    pc = geo.instances_pointcloud()
    matrices = db.Attribute(pc.attributes["instance_transform"]).as_array()
    np.testing.assert_allclose(matrices[:, 3, :3], q[:, :3], atol=1e-5)
    np.testing.assert_allclose(
        np.linalg.norm(matrices[:, :3, :3], axis=2), 0.5, atol=1e-4
    )
    assert pc.attributes.get("collider") is not None, "source attributes carried over"

    # Re-compiling keeps the output object and its modifier
    obj.modifiers.new("Keep Me", "NODES")
    sim.compile()
    assert sim.instances.output.object == obj
    assert "Keep Me" in obj.modifiers


def test_instances_land_on_rigid_collection_box():
    """Instance bodies come after the rigid collection's in the model; a
    kinematic box still holds its place and catches them."""
    sim = _small_monkeys((2, 2, 1))
    collection = bpy.data.collections.new("Warbler Rigid")
    get_scene().collection.children.link(collection)
    demos._rigid_box("Table", collection, (0.0, 0.0, 0.5), (8.0, 8.0, 1.0))
    sim.props.sim_rigid_collection = collection
    q = _run_bodies(sim, 120)
    assert sim.model.body_count == 5
    np.testing.assert_allclose(q[0, :3], (0.0, 0.0, 0.5), atol=1e-5)
    assert q[1:, 2].min() > 1.0, "monkeys rest on the table"


def test_instance_velocity():
    """An INSTANCE ``velocity`` attribute sets each body's starting velocity."""
    with g.tree("Warbler Thrown Monkey", clear=True, is_modifier=True) as tree:
        (
            g.InstanceOnPoints(
                g.Points(1, position=(0.0, 0.0, 3.0)),
                instance=g.ObjectInfo(demos._monkey()).o.geometry,
                scale=0.5,
            )
            >> g.StoreNamedAttribute.instance.vector(
                name="velocity", value=(4.0, 0.0, 0.0)
            )
            >> tree.outputs.geometry("Geometry")
        )
    source = demos._source_object("ThrownMonkey", tree.tree)
    sim = demos._new_simulation(bpy.context, source)
    sim.props.solver = "XPBD"
    q = _run_bodies(sim, 10)
    # 10 frames at 60 fps: x = 4 m/s * 1/6 s
    assert q[0, 0] == pytest.approx(4.0 / 6.0, rel=0.05)


def _floating_monkeys(solver: str, drag: float, temperature: float) -> Simulator:
    """Monkeys spread out in zero gravity, with no ground to hit."""
    sim = _small_monkeys((3, 3, 3))
    source = sim.props.source
    assert source is not None
    demos.set_modifier_input(source, "Spacing", 4.0)
    sim.props.solver = solver
    sim.props.use_ground_plane = False
    sim.props.gravity = 0.0
    sim.props.drag = drag
    sim.props.temperature = temperature
    return sim


@pytest.mark.parametrize("solver", ["XPBD", "VBD"])
def test_rigid_bodies_feel_drag(solver):
    """Drag slows a moving body as exp(-drag·t), linear and angular."""
    sim = _floating_monkeys(solver, drag=2.0, temperature=0.0)
    get_scene().frame_set(1)
    sim.compile()
    qd = np.zeros((sim.model.body_count, 6), dtype=np.float32)
    qd[:, 0] = 3.0
    qd[:, 5] = 2.0
    sim.state_0.body_qd.assign(qd)  # ty: ignore[unresolved-attribute]
    for f in range(2, 32):
        get_scene().frame_set(f)
    qd = sim.state_0.body_qd.numpy()  # ty: ignore[unresolved-attribute]
    # 30 frames at 60 fps
    decay = np.exp(-2.0 * 0.5)
    np.testing.assert_allclose(qd[:, 0], 3.0 * decay, rtol=0.02)
    assert np.linalg.norm(qd[:, 3:], axis=1).max() < 2.0 * decay * 1.05


@pytest.mark.parametrize("solver", ["XPBD", "VBD"])
def test_rigid_bodies_feel_temperature(solver):
    """The thermostat holds rigid bodies at equipartition: <m·v²> = kT per
    axis and <ω·I·ω> = 3·kT, with ω and I in the body frame."""
    kT = 10.0
    sim = _floating_monkeys(solver, drag=2.0, temperature=kT)
    get_scene().frame_set(1)
    sim.compile()
    mass = sim.model.body_mass.numpy()  # ty: ignore[unresolved-attribute]
    inertia = sim.model.body_inertia.numpy()  # ty: ignore[unresolved-attribute]
    linear, angular = [], []
    for f in range(2, 302):
        get_scene().frame_set(f)
        if f < 100:
            continue  # equilibrating
        q = sim.state_0.body_q.numpy()  # ty: ignore[unresolved-attribute]
        qd = sim.state_0.body_qd.numpy()  # ty: ignore[unresolved-attribute]
        rot = Rotation.from_quat(q[:, 3:7]).as_matrix()
        w = np.einsum("nji,nj->ni", rot, qd[:, 3:])
        linear.append((mass[:, None] * qd[:, :3] ** 2).mean())
        angular.append(np.einsum("ni,nij,nj->n", w, inertia, w).mean())
    assert np.mean(linear) == pytest.approx(kT, rel=0.15)
    assert np.mean(angular) == pytest.approx(3 * kT, rel=0.15)


GROUPS = [getattr(nodes.geometry, name) for name in nodes.geometry.__all__]


@pytest.mark.parametrize("group", GROUPS, ids=lambda c: c._name)
def test_node_groups_build_from_source(group):
    """Every group's recipe builds (as `nodebpy build` does) and is reused."""
    with build_from_source():
        tree = group.create_group()
        assert tree.name == group._name
        assert group.create_group() == tree


def test_asset_library():
    """The built library holds every group as an asset in a Warbler catalog,
    and is registered as a Blender asset library."""
    path = nodes.library.NODES_FILE
    if not path.exists():
        pytest.skip("asset library not built (uv run nodebpy build)")
    catalogs = {
        line.split(":")[0]: line.split(":")[1]
        for line in (path.parent / "blender_assets.cats.txt").read_text().splitlines()
        if line and not line.startswith(("#", "VERSION"))
    }
    load = bpy.data.libraries.load(str(path), assets_only=True)
    with load as (src, _):  # ty: ignore[invalid-context-manager]
        assert sorted(src.node_groups) == sorted(c._asset_name for c in GROUPS)
    for group in GROUPS:
        tree = group.create_group()  # linked from the library
        assert tree.library is not None
        assert tree.asset_data is not None
        assert catalogs[tree.asset_data.catalog_id].startswith("Warbler/")
    preferences = bpy.context.preferences
    assert preferences is not None
    library = preferences.filepaths.asset_libraries[nodes.library.LIBRARY_NAME]
    assert library.path == str(path.parent)  # ty: ignore[unresolved-attribute]


@pytest.mark.parametrize("shape", SHAPES)
def test_set_collider_writes_the_shape_index(shape):
    """Set Collider's menu maps onto the ``collider`` values warbler reads."""
    with g.tree("Collider Test", clear=True, is_modifier=True) as tree:
        (
            g.InstanceOnPoints(g.Points(3), instance=g.Cube())
            >> nodes.geometry.SetCollider(shape=shape)
            >> tree.outputs.geometry("Geometry")
        )
    source = demos._source_object("ColliderTest", tree.tree)
    collider = GeometrySet(source).instances.to_props()["collider"]
    assert (collider == COLLIDERS.index(shape.upper().replace(" ", "_"))).all()


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
