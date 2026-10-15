"""Demo scenes: Newton examples rebuilt as Geometry Nodes sources.

Each demo builds a node tree with nodebpy that generates the geometry and tags
it with the attributes warbler reads (``mass``, ``radius``, ``pinned``,
``velocity``), puts it on a source object, and configures a simulation to
match the Newton example it is modelled on.
"""

import math
from collections.abc import Callable

import bpy
import databpy as db
import numpy as np
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


def set_modifier_input(obj: bpy.types.Object, name: str, value) -> None:
    """Set an input of the source's GN modifier by its interface name."""
    mod = obj.modifiers["Warbler Source"]
    assert isinstance(mod, bpy.types.NodesModifier) and mod.node_group is not None
    interface = mod.node_group.interface
    assert interface is not None
    item = interface.items_tree[name]
    getattr(mod.properties.inputs, item.identifier).value = value  # ty: ignore[unresolved-attribute]
    obj.update_tag()


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


def hanging_cloth_tree(
    name: str = "Warbler Hanging Cloth", edge_attribute: str = "pinned"
) -> bpy.types.GeometryNodeTree:
    """A cloth grid pinned along one edge, as ``add_cloth_grid(fix_left=True)``.

    Newton builds the grid with its corner at ``pos`` and rotates it 90° about
    Z, so the pinned (left) edge lies along world Y at x = 0. The edge is
    tagged with ``edge_attribute`` (``attach`` holds it to a rigid body instead).
    """
    with g.tree(name, clear=True, is_modifier=True) as tree:
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
            >> g.StoreNamedAttribute.point.boolean(name=edge_attribute, value=pinned)
            >> g.StoreNamedAttribute.point.float(name="mass", value=mass)
            >> g.StoreNamedAttribute.point.float(name="radius", value=radius)
            >> tree.outputs.geometry("Geometry")
        )
    return tree.tree


def cloth(context: bpy.types.Context) -> Simulator:
    """Cloth hanging from one edge, swinging down under gravity."""
    source = _source_object("HangingCloth", hanging_cloth_tree())
    sim = _new_simulation(context, source)
    _cloth_settings(sim.props)
    return sim


def _cloth_settings(item: SimulationListItem) -> None:
    """example_cloth_hanging's VBD cloth parameters."""
    item.solver = "VBD"
    item.cloth_tri_ke = 1.0e3
    item.cloth_tri_ka = 1.0e3
    item.cloth_tri_kd = 1.0e2
    item.cloth_edge_ke = 1.0e1
    item.cloth_edge_kd = 0.0
    item.soft_contact_ke = 1.0e2
    item.soft_contact_kd = 1.0e2
    item.soft_contact_mu = 1.0


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


# ============================================================================
# Membrane fusion — lipids held on a remeshed surface
# ============================================================================


def fusing_vesicles_tree() -> bpy.types.GeometryNodeTree:
    """Two spherical vesicles that fuse between two frames.

    The membrane is the zero level set of a smooth union of two sphere SDFs,
    remeshed every frame: as the centres approach and the smoothing grows, a
    neck forms and widens until one vesicle is left.
    """
    with g.tree("Warbler Fusing Vesicles", clear=True, is_modifier=True) as tree:
        radius = tree.inputs.float("Radius", 1.0, min_value=0.01)
        gap = tree.inputs.float("Gap", 0.2, min_value=0.0)
        start = tree.inputs.float("Start Frame", 30.0)
        end = tree.inputs.float("End Frame", 330.0)
        overlap = tree.inputs.float("Overlap", 0.6, min_value=0.0, max_value=1.0)
        smoothing = tree.inputs.float("Smoothing", 0.6, min_value=0.0)
        resolution = tree.inputs.integer("Resolution", 48, min_value=8)

        fusion = g.MapRange.float(
            g.SceneTime().o.frame, start, end, interpolation_type="SMOOTHSTEP"
        )
        apart = radius + gap / 2
        offset = apart + (radius * (1 - overlap) - apart) * fusion
        centre = g.CombineXYZ(x=offset)
        # Grow the spheres as they overlap so the membrane keeps its area
        # (4πr(r + d) for two spheres of radius r with centres ±d), else the
        # lipids end up packed onto a ~30% smaller surface.
        grown = g.Math.maximum(
            radius,
            (g.Math.square_root(offset * offset + radius * radius * 8) - offset) / 2,
        )
        position = g.Position().o.position
        sdf = g.Math.smooth_minimum(
            g.VectorMath.length(position - centre) - grown,
            g.VectorMath.length(position + centre) - grown,
            fusion * smoothing * radius,
        )
        # Bounds of the separated vesicles and the grown fused one, plus a
        # margin for the remesh
        half = radius * 1.4
        # Background is "outside", so the cube's walls don't make a surface
        volume = g.VolumeCube(
            density=-sdf,
            background=-1.0,
            min=g.CombineXYZ(-(apart + half), -half, -half),
            max=g.CombineXYZ(apart + half, half, half),
            resolution_x=resolution * 2,
            resolution_y=resolution,
            resolution_z=resolution,
        )
        (
            g.VolumeToMesh(volume, threshold=0.0, adaptivity=0.0)
            >> g.SetShadeSmooth()
            >> tree.outputs.geometry("Geometry")
        )
    return tree.tree


def membrane_lipids_tree(surface: bpy.types.Object) -> bpy.types.GeometryNodeTree:
    """Lipid points spread evenly over ``surface``, tagged ``on_surface``.

    The scatter is held in a simulation zone, so it's made once on the first
    frame from the real surface and not re-run as the surface changes (a
    Poisson scatter every frame costs ~40 ms). ``vesicle`` records which
    vesicle each lipid started on, so their mixing after fusion can be
    coloured.
    """
    with g.tree("Warbler Membrane Lipids", clear=True, is_modifier=True) as tree:
        radius = tree.inputs.float("Radius", 0.044, min_value=0.001)
        mass = tree.inputs.float("Mass", 1.0, min_value=0.0)

        scatter = g.DistributePointsOnFaces(
            g.ObjectInfo(surface).o.geometry,
            distance_min=radius,
            density_max=1.0e4,
            distribute_method="POISSON",
        )
        cache = g.SimulationZone()
        lipids = cache.items.geometry(scatter, "Lipids")
        lipids.current >> lipids.next
        (
            lipids.result
            >> g.StoreNamedAttribute.point.float(name="radius", value=radius)
            >> g.StoreNamedAttribute.point.float(name="mass", value=mass)
            >> g.StoreNamedAttribute.point.boolean(name="on_surface", value=True)
            >> g.StoreNamedAttribute.point.integer(
                name="vesicle", value=g.Position().o.position.x > 0
            )
            >> tree.outputs.geometry("Geometry")
        )
    return tree.tree


def _attribute_material(name: str, attribute_name: str) -> bpy.types.Material:
    """Blue where ``attribute_name`` is 0, orange where it is 1."""
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    tree = mat.node_tree
    assert tree is not None
    bsdf = tree.nodes["Principled BSDF"]
    attribute = tree.nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = attribute_name  # ty: ignore[unresolved-attribute]
    mix = tree.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"  # ty: ignore[unresolved-attribute]
    mix.inputs[6].default_value = (0.1, 0.45, 0.9, 1.0)  # ty: ignore[unresolved-attribute]
    mix.inputs[7].default_value = (0.95, 0.45, 0.1, 1.0)  # ty: ignore[unresolved-attribute]
    tree.links.new(attribute.outputs["Fac"], mix.inputs["Factor"])
    tree.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    return mat


def membrane_fusion(context: bpy.types.Context) -> Simulator:
    """Lipids diffusing on two vesicles as they fuse (XPBD + surface projection)."""
    surface = _source_object("FusingVesicles", fusing_vesicles_tree())
    source = _source_object("MembraneLipids", membrane_lipids_tree(surface))
    # The output is matched by name on compile, so the colour survives.
    name = f"{source.name}_points"
    output = bpy.data.objects.get(name)
    if output is None:
        output = bpy.data.objects.new(name, bpy.data.pointclouds.new(name))
        get_scene().collection.objects.link(output)
    output.data.materials.clear()  # ty: ignore[unresolved-attribute]
    # Colour lipids by the vesicle they started on
    output.data.materials.append(_attribute_material("Warbler Vesicle", "vesicle"))  # ty: ignore[unresolved-attribute]

    sim = _new_simulation(context, source)
    item = sim.props
    item.solver = "XPBD"
    item.surface = surface
    item.gravity = 0.0
    item.use_ground_plane = False
    item.temperature = 0.02
    item.drag = 5.0
    get_scene().frame_end = 400
    return sim


# ============================================================================
# Carried cloth — particle-body attachments and contact outputs (VBD)
# ============================================================================


def _rigid_box(
    name: str,
    collection: bpy.types.Collection,
    location: tuple[float, float, float],
    size: tuple[float, float, float],
) -> bpy.types.Object:
    """A box mesh object in ``collection``, for use as a rigid body."""
    obj = bpy.data.objects.get(name)
    if obj is None:
        mesh = bpy.data.meshes.new(name)
        half = np.array(size) / 2
        signs = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        faces = [
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        ]
        mesh.from_pydata((signs * half).tolist(), [], faces)
        obj = bpy.data.objects.new(name, mesh)
    for c in obj.users_collection:
        c.objects.unlink(obj)
    collection.objects.link(obj)
    obj.animation_data_clear()
    obj.location = location
    return obj


def carried_cloth(context: bpy.types.Context) -> Simulator:
    """A sheet attached to an animated bar, dragged over a box (VBD).

    The cloth's edge is held by ``attach`` (particle-body springs) rather than
    pins, and its output is coloured by ``is_colliding``.
    """
    tree = hanging_cloth_tree("Warbler Carried Cloth", edge_attribute="attach")
    source = _source_object("CarriedCloth", tree)
    for name, value in (("Dim X", 32), ("Dim Y", 32), ("Height", 3.0)):
        set_modifier_input(source, name, value)

    scene = get_scene()
    collection = bpy.data.collections.get("Warbler Rigid")
    if collection is None:
        collection = bpy.data.collections.new("Warbler Rigid")
        scene.collection.children.link(collection)
    # The cloth edge runs along x in [-3.2, 0] at y = 0. The bar sits just
    # clear of it: attached particles inside the body would also collide with it.
    bar = _rigid_box("Bar", collection, (-1.6, -0.15, 3.0), (3.4, 0.1, 0.1))
    bar.keyframe_insert("location", frame=1)
    bar.location = (-1.6, 4.4, 3.0)
    bar.keyframe_insert("location", frame=180)
    _rigid_box("Obstacle", collection, (-1.6, 2.5, 0.75), (1.5, 1.5, 1.5))

    # The output is matched by name on compile, so the colour survives.
    name = f"{source.name}_mesh"
    output = bpy.data.objects.get(name)
    if output is None:
        output = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        scene.collection.objects.link(output)
    output.data.materials.clear()  # ty: ignore[unresolved-attribute]
    output.data.materials.append(  # ty: ignore[unresolved-attribute]
        _attribute_material("Warbler Colliding", "is_colliding")
    )

    sim = _new_simulation(context, source)
    _cloth_settings(sim.props)
    sim.props.sim_rigid_collection = collection
    scene.frame_end = 240
    return sim


# ============================================================================
# Protein — elastic network model on a Molecular Nodes molecule
# ============================================================================


# Residues either side averaged when smoothing pLDDT confidence, so short dips
# inside a domain don't split it in two.
PLDDT_WINDOW = 3


def protein_network_tree(molecule: bpy.types.Object) -> bpy.types.GeometryNodeTree:
    """Alpha carbons of one chain as beads on a backbone, tagged for the network.

    Beads keep every Molecular Nodes attribute and their exact atom positions.
    Consecutive residues are joined by mesh edges (permanent backbone bonds,
    "Backbone Stiffness" times stiffer than contacts so pulling unfolds the
    protein rather than tearing the chain); contacts come from the
    simulation's network cutoff. Beads are picked with
    the ``is_alpha_carbon`` attribute Molecular Nodes stores on atoms (read
    from the "MN/Atoms" bundle when a style is applied), so this
    doesn't need to import Molecular Nodes (which as a Blender extension isn't
    importable as ``molecularnodes``).

    With "Domains from pLDDT", AlphaFold's per-residue confidence (stored in
    ``b_factor``) is smoothed over ±``PLDDT_WINDOW`` residues; each confident
    run becomes a ``domain`` and low-confidence residues get ``domain`` -1 and
    a soft ``stiffness``. "Pull" moves the pinned termini apart along the line
    between them, for unfolding the protein live.
    """
    with g.tree("Warbler Protein Network", clear=True, is_modifier=True) as tree:
        chain = tree.inputs.integer("Chain", 0, min_value=0)
        mass = tree.inputs.float("Mass", 1.0, min_value=0.0)
        radius = tree.inputs.float("Radius", 0.14, min_value=0.001)
        use_plddt = tree.inputs.boolean("Domains from pLDDT", True)
        threshold = tree.inputs.float("pLDDT Threshold", 70.0)
        soft = tree.inputs.float("Disordered Stiffness", 0.05, min_value=0.0)
        bond_stiffness = tree.inputs.float("Backbone Stiffness", 20.0, min_value=0.0)
        pin_termini = tree.inputs.boolean("Pin Termini", True)
        pull = tree.inputs.float("Pull", 0.0)

        # When the molecule's output carries its atoms in the geometry bundle
        # under "MN/Atoms" (as Molecular Nodes' "Evaluate on Atoms" writes and
        # "Get Geometry Atoms" reads), use those; otherwise the geometry is the
        # atoms. Object Info's transform isn't applied to bundle contents.
        info = g.ObjectInfo(molecule, transform_space="RELATIVE")
        bundle = g.GetGeometryBundle(info.o.geometry)
        bundled = g.GetBundleItem.geometry(bundle.o.bundle, "MN/Atoms")
        bundled_atoms = g.TransformGeometry(
            bundled.o.item, mode="Matrix", transform=info.o.transform
        )
        atoms = bundled.o.exists.switch.geometry(bundle.o.geometry, bundled_atoms)
        in_chain = g.NamedAttribute.integer("chain_id") == chain
        is_ca = g.NamedAttribute.boolean("is_alpha_carbon")
        backbone = (
            g.SeparateGeometry.point(atoms, is_ca & in_chain).o.selection
            >> g.MeshToPoints()
            >> g.PointsToCurves(weight=g.NamedAttribute.integer("res_id"))
            >> g.CurveToMesh()
        )

        idx = g.Index()
        last = g.DomainSize(backbone).o.point_count - 1

        # Confident residues, smoothed by a majority vote over the window
        plddt = g.Math.greater_than(
            g.NamedAttribute.float("b_factor"), threshold
        ).o.value
        votes = plddt.point.at(idx - PLDDT_WINDOW)
        for k in range(-PLDDT_WINDOW + 1, PLDDT_WINDOW + 1):
            votes = votes + plddt.point.at(idx + k)
        confident = use_plddt.switch.boolean(True, votes > PLDDT_WINDOW + 0.5)

        # Number each confident run: a run starts where the previous bead isn't
        previous = confident.point.at(idx - 1)
        start = confident & (~previous | (idx == 0))
        run = g.AccumulateField.point.integer(start).o.leading
        domain = confident.switch.integer(-1, run - 1)
        stiffness = confident.switch.float(soft, 1.0)

        # Pinned termini, pulled apart along the line between them
        position = g.Position().o.position
        first_pos = position.point.at(0)
        last_pos = position.point.at(last)
        direction = (last_pos - first_pos).normalize()
        is_last = idx == last
        is_end = (idx == 0) | is_last
        sign = g.Switch.float(is_last, -0.5, 0.5)
        pulled = g.SetPosition(
            backbone, selection=is_end, offset=direction * (pull * sign)
        )

        (
            pulled
            >> g.StoreNamedAttribute.point.boolean(
                name="pinned", value=pin_termini & is_end
            )
            >> g.StoreNamedAttribute.point.integer(name="domain", value=domain)
            >> g.StoreNamedAttribute.point.float(name="stiffness", value=stiffness)
            >> g.StoreNamedAttribute.edge.float(
                name="bond_stiffness", value=bond_stiffness
            )
            >> g.StoreNamedAttribute.point.float(name="mass", value=mass)
            >> g.StoreNamedAttribute.point.float(name="radius", value=radius)
            >> tree.outputs.geometry("Geometry")
        )
    return tree.tree


def has_plddt(molecule: bpy.types.Object) -> bool:
    """Whether the molecule's ``b_factor`` looks like AlphaFold pLDDT.

    AlphaFold writes one pLDDT per residue to every atom in it; crystal
    B-factors vary between atoms of a residue (backbone vs side chain).
    """
    b = db.named_attribute(molecule, "b_factor")
    residue = db.named_attribute(molecule, "chain_id").astype(
        np.int64
    ) * 1_000_000 + db.named_attribute(molecule, "res_id")
    _, inverse = np.unique(residue, return_inverse=True)
    high = np.full(inverse.max() + 1, -np.inf)
    low = np.full(inverse.max() + 1, np.inf)
    np.maximum.at(high, inverse, b)
    np.minimum.at(low, inverse, b)
    return bool(np.mean(high - low < 1e-3) > 0.95)


def protein_network(
    context: bpy.types.Context,
    molecule: bpy.types.Object,
    plddt: bool | None = None,
) -> Simulator:
    """A floating protein held together by an elastic network, jiggling thermally.

    Lengths assume Molecular Nodes' default scale of 0.1 Blender units per Å:
    an 8 Å network cutoff and 1.4 Å bead radius. ``plddt`` turns on domains
    from AlphaFold confidence; by default it's detected with ``has_plddt``.
    """
    source = _source_object(f"{molecule.name}_network", protein_network_tree(molecule))
    if plddt is None:
        plddt = has_plddt(molecule)
    set_modifier_input(source, "Domains from pLDDT", plddt)
    sim = _new_simulation(context, source)
    item = sim.props
    item.solver = "XPBD"
    item.gravity = 0.0
    item.use_ground_plane = False
    item.network_cutoff = 0.8
    item.spring_ke = 1.0e3
    item.spring_kd = 1.0
    item.temperature = 10.0
    item.drag = 5.0
    item.domain_coupling = 0.1
    item.break_strain = 0.5
    item.reform_strain = 0.25
    return sim


DEMOS: dict[str, tuple[str, str, Callable[[bpy.types.Context], Simulator]]] = {
    "PARTICLES": ("Particle Grid", "Jittered block of particles (XPBD)", particles),
    "CLOTH": ("Hanging Cloth", "Newton's example_cloth_hanging (VBD)", cloth),
    "ROPES": (
        "Hanging Ropes",
        "Curves as spring chains with animated pins (XPBD)",
        ropes,
    ),
    "CARRIED_CLOTH": (
        "Carried Cloth",
        "Cloth attached to an animated bar, coloured by contact (VBD)",
        carried_cloth,
    ),
    "MEMBRANE_FUSION": (
        "Membrane Fusion",
        "Lipids held on two fusing vesicles by surface projection (XPBD)",
        membrane_fusion,
    ),
}
