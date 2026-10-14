"""Demo scenes: Newton examples rebuilt as Geometry Nodes sources.

Each demo builds a node tree with nodebpy that generates the geometry and tags
it with the attributes warbler reads (``mass``, ``radius``, ``pinned``,
``velocity``), puts it on a source object, and configures a simulation to
match the Newton example it is modelled on.
"""

import math
from collections.abc import Callable

import bpy
from nodebpy import geometry as g

from .manager import get_manager
from .props import SimulationListItem
from .simulation import Simulator
from .utils import get_scene


def _source_object(name: str, tree: bpy.types.GeometryNodeTree) -> bpy.types.Object:
    """An empty mesh object whose geometry comes entirely from ``tree``."""
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        get_scene().collection.objects.link(obj)
    obj.modifiers.clear()
    mod = obj.modifiers.new("Warbler Source", "NODES")
    assert isinstance(mod, bpy.types.NodesModifier)
    mod.node_group = tree
    # The simulation writes to separate output objects; keep the source as a
    # wireframe reference so it doesn't hide them.
    obj.display_type = "WIRE"
    return obj


def _new_simulation(context: bpy.types.Context, source: bpy.types.Object) -> Simulator:
    """Add a simulation reading from ``source`` with Newton's example timing."""
    # Newton's examples run at 60 fps with 10 substeps and 10 iterations.
    scene = get_scene()
    scene.render.fps = 60
    scene.render.fps_base = 1.0
    man = get_manager(context)
    sim = Simulator()
    man.add(sim)
    man.item_index = len(man.sim_items) - 1
    item: SimulationListItem = sim.props
    item.source = source
    item.substeps = 10
    item.iterations = 10
    return sim


# ============================================================================
# Particles — add_particle_grid() from example_xpbd_vbd_coupled_solver
# ============================================================================


def particle_grid_tree() -> bpy.types.GeometryNodeTree:
    """A jittered 3D grid of points, as ``ModelBuilder.add_particle_grid``."""
    with g.tree("Warbler Particle Grid", clear=True, is_modifier=True) as tree:
        dim_x = tree.inputs.integer("Dim X", 16, min_value=1)
        dim_y = tree.inputs.integer("Dim Y", 16, min_value=1)
        dim_z = tree.inputs.integer("Dim Z", 16, min_value=1)
        cell = tree.inputs.float("Cell Size", 0.05, min_value=0.001)
        origin = tree.inputs.vector("Origin", (-0.12, -0.12, 0.66))
        radius = tree.inputs.float("Radius", 0.025, min_value=0.001)
        mass = tree.inputs.float("Mass", 0.02, min_value=0.0)
        jitter = tree.inputs.float("Jitter", 0.002, min_value=0.0)

        idx = g.Index()
        cell_index = (idx % dim_x, (idx // dim_x) % dim_y, idx // (dim_x * dim_y))
        offset = g.RandomValue.vector(min=-jitter, max=jitter, seed=1)
        position = origin + g.CombineXYZ(*cell_index) * cell + offset

        (
            g.Points(dim_x * dim_y * dim_z, position=position, radius=radius)
            >> g.StoreNamedAttribute.point.float(name="mass", value=mass)
            >> tree.outputs.geometry("Geometry")
        )
    return tree.tree


def particles(context: bpy.types.Context) -> Simulator:
    """A block of particles falling onto the ground (XPBD granular contact)."""
    source = _source_object("ParticleGrid", particle_grid_tree())
    sim = _new_simulation(context, source)
    sim.props.solver = "XPBD"
    return sim


# ============================================================================
# Cloth — example_cloth_hanging (VBD)
# ============================================================================


def hanging_cloth_tree() -> bpy.types.GeometryNodeTree:
    """A cloth grid pinned along one edge, as ``add_cloth_grid(fix_left=True)``.

    Newton builds the grid with its corner at ``pos`` and rotates it 90° about
    Z, so the pinned (left) edge lies along world Y at x = 0.
    """
    with g.tree("Warbler Hanging Cloth", clear=True, is_modifier=True) as tree:
        dim_x = tree.inputs.integer("Dim X", 64, min_value=1)
        dim_y = tree.inputs.integer("Dim Y", 32, min_value=1)
        cell = tree.inputs.float("Cell Size", 0.1, min_value=0.001)
        height = tree.inputs.float("Height", 4.0)
        mass = tree.inputs.float("Mass", 0.1, min_value=0.0)
        radius = tree.inputs.float("Radius", 0.05, min_value=0.001)

        size_x = dim_x * cell
        size_y = dim_y * cell
        grid = g.Grid(size_x, size_y, dim_x + 1, dim_y + 1)
        # Grid is centred; move its corner to the origin before rotating.
        corner = g.TransformGeometry(grid, translation=(size_x / 2, size_y / 2, 0))  # ty: ignore[invalid-argument-type]
        cloth = g.TransformGeometry(
            corner,
            translation=(0, 0, height),  # ty: ignore[invalid-argument-type]
            rotation=(0, 0, math.pi / 2),
        )
        pinned = g.Position().o.position.y < cell * 0.01

        (
            cloth
            >> g.StoreNamedAttribute.point.boolean(name="pinned", value=pinned)
            >> g.StoreNamedAttribute.point.float(name="mass", value=mass)
            >> g.StoreNamedAttribute.point.float(name="radius", value=radius)
            >> tree.outputs.geometry("Geometry")
        )
    return tree.tree


def cloth(context: bpy.types.Context) -> Simulator:
    """Cloth hanging from one edge, swinging down under gravity."""
    source = _source_object("HangingCloth", hanging_cloth_tree())
    sim = _new_simulation(context, source)
    item = sim.props
    item.solver = "VBD"
    item.cloth_tri_ke = 1.0e3
    item.cloth_tri_ka = 1.0e3
    item.cloth_tri_kd = 1.0e2
    item.cloth_edge_ke = 1.0e1
    item.cloth_edge_kd = 0.0
    item.soft_contact_ke = 1.0e2
    item.soft_contact_kd = 1.0e2
    item.soft_contact_mu = 1.0
    return sim


# ============================================================================
# Curves — ropes as particle chains with animated pins
# ============================================================================


def hanging_ropes_tree() -> bpy.types.GeometryNodeTree:
    """A row of horizontal ropes, pinned at their first point.

    The pinned roots sway along Y over time; warbler re-reads pinned positions
    every frame so the ropes are dragged along live.
    """
    with g.tree("Warbler Hanging Ropes", clear=True, is_modifier=True) as tree:
        count = tree.inputs.integer("Count", 8, min_value=1)
        length = tree.inputs.float("Length", 2.0, min_value=0.01)
        resolution = tree.inputs.integer("Resolution", 32, min_value=2)
        spacing = tree.inputs.float("Spacing", 0.3, min_value=0.0)
        height = tree.inputs.float("Height", 2.5)
        radius = tree.inputs.float("Radius", 0.02, min_value=0.001)
        mass = tree.inputs.float("Mass", 0.05, min_value=0.0)
        sway = tree.inputs.float("Sway", 0.5)

        roots = g.Points(
            count,
            position=(0, (g.Index() - (count - 1) / 2) * spacing, height),  # ty: ignore[invalid-argument-type]
        )
        line = g.CurveLine(start=(0, 0, 0), end=(length, 0, 0))  # ty: ignore[invalid-argument-type]
        ropes = (
            g.InstanceOnPoints(roots, instance=line)
            >> g.RealizeInstances()
            >> g.ResampleCurve(count=resolution)
        )
        pinned = g.EndpointSelection(start_size=1, end_size=0)
        time = g.SceneTime().o.seconds
        swayed = g.SetPosition(
            ropes,
            selection=pinned,
            offset=(0, g.Math.sine(time * 2) * sway, 0),  # ty: ignore[invalid-argument-type]
        )

        (
            swayed
            >> g.StoreNamedAttribute.point.boolean(name="pinned", value=pinned)
            >> g.StoreNamedAttribute.point.float(name="mass", value=mass)
            >> g.SetCurveRadius(radius=radius)
            >> tree.outputs.geometry("Geometry")
        )
    return tree.tree


def ropes(context: bpy.types.Context) -> Simulator:
    """Ropes swinging down from roots that are animated by Geometry Nodes."""
    source = _source_object("HangingRopes", hanging_ropes_tree())
    sim = _new_simulation(context, source)
    item = sim.props
    item.solver = "XPBD"
    item.spring_ke = 1.0e3
    item.spring_kd = 1.0
    item.bend_ke = 1.0e1
    return sim


DEMOS: dict[str, tuple[str, str, Callable[[bpy.types.Context], Simulator]]] = {
    "PARTICLES": ("Particle Grid", "Jittered block of particles (XPBD)", particles),
    "CLOTH": ("Hanging Cloth", "Newton's example_cloth_hanging (VBD)", cloth),
    "ROPES": (
        "Hanging Ropes",
        "Curves as spring chains with animated pins (XPBD)",
        ropes,
    ),
}
