"""Elastic network model of a protein from a Molecular Nodes molecule."""

import itertools
from pathlib import Path

import bpy
import databpy as db
import molecularnodes as mn
import numpy as np
import pytest

from warbler import demos
from warbler.manager import get_manager
from warbler.simulation import Simulator, network_edges
from warbler.utils import get_scene

# Structures are downloaded on first use and kept here (gitignored)
CACHE = Path(__file__).parent / ".cache"


def fetch(code: str, database: str = "rcsb") -> bpy.types.Object:
    return mn.Molecule.fetch(code, cache=CACHE, database=database).object


mn.register()


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD after optimally superimposing ``b`` onto ``a``."""
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(b0.T @ a0)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((b0 @ r - a0) ** 2).sum(1).mean()))


@pytest.fixture
def molecule() -> bpy.types.Object:
    """Adenylate kinase (4AKE), loaded with Molecular Nodes."""
    return fetch("4AKE")


def _chain_a_alpha_carbons(obj: bpy.types.Object) -> dict[str, np.ndarray]:
    mask = db.named_attribute(obj, "is_alpha_carbon") & (
        db.named_attribute(obj, "chain_id") == 0
    )
    return {
        name: db.named_attribute(obj, name)[mask]
        for name in ("position", "res_id", "b_factor")
    }


def test_network_edges_matches_brute_force():
    rng = np.random.default_rng(0)
    pos = rng.random((300, 3))
    edges = network_edges(pos, 0.15)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    i, j = np.nonzero(np.triu(d < 0.15, k=1))
    assert {tuple(e) for e in edges.tolist()} == set(zip(i.tolist(), j.tolist()))


def test_protein_network_builds(molecule):
    """Beads are exactly chain A's alpha carbons, carrying their MN attributes."""
    sim = demos.protein_network(bpy.context, molecule)
    sim.compile()
    ca = _chain_a_alpha_carbons(molecule)

    assert len(sim.parts) == 1
    part = sim.parts[0]
    assert part.count == len(ca["position"]) == 214
    np.testing.assert_allclose(sim.particle_positions, ca["position"], atol=1e-5)

    # Every spring is a pair within the cutoff, and every such pair is a spring
    assert part.edges is not None
    expected = network_edges(ca["position"], sim.props.network_cutoff)
    assert sim.model.spring_count == len(part.edges) == len(expected)
    lengths = np.linalg.norm(
        ca["position"][part.edges[:, 0]] - ca["position"][part.edges[:, 1]], axis=1
    )
    assert lengths.max() < sim.props.network_cutoff
    # ...including the whole backbone: consecutive CAs are ~3.8 Å apart
    pairs = {tuple(sorted(e)) for e in part.edges.tolist()}
    order = np.argsort(ca["res_id"])
    assert all(
        (int(a), int(b)) in pairs or (int(b), int(a)) in pairs
        for a, b in itertools.pairwise(order)
    )

    # The output is a mesh of the network with MN's attributes intact
    out = part.output
    assert out is not None and out.object.type == "MESH"
    assert len(out.object.data.edges)  # ty: ignore[unresolved-attribute] == len(part.edges)
    np.testing.assert_array_equal(out.named_attribute("res_id"), ca["res_id"])
    np.testing.assert_allclose(out.named_attribute("b_factor"), ca["b_factor"])
    assert (out.named_attribute("chain_id") == 0).all()


def test_protein_holds_together_and_jiggles(molecule):
    sim = demos.protein_network(bpy.context, molecule)
    sim.props.device = "cpu"
    scene = get_scene()
    scene.frame_set(1)
    sim.compile()
    q0 = sim.particle_positions.copy()
    edges = sim.parts[0].edges
    assert edges is not None
    rest = np.linalg.norm(q0[edges[:, 0]] - q0[edges[:, 1]], axis=1)

    for f in range(2, 122):
        scene.frame_set(f)
    assert sim.props.is_active
    q = sim.particle_positions
    assert np.isfinite(q).all()

    # Thermal motion is visible (> 0.2 Å) but the fold holds (< 3 Å)
    rmsd = _kabsch_rmsd(q0, q)
    assert 0.02 < rmsd < 0.3, rmsd
    strain = np.abs(np.linalg.norm(q[edges[:, 0]] - q[edges[:, 1]], axis=1) / rest - 1)
    assert strain.max() < 0.25


def test_thermostat_holds_temperature():
    """Free particles under the Langevin thermostat reach <v²> = kT/m per axis."""
    n, mass, kT = 2000, 2.0, 3.0
    grid = np.stack(np.meshgrid(*[np.arange(13)] * 3), -1).reshape(-1, 3)[:n] * 1.0
    source = db.BlenderObject.from_pointcloud(grid, name="Gas")
    source.store_named_attribute(np.full(n, mass, dtype=np.float32), "mass")
    source.store_named_attribute(np.full(n, 0.01, dtype=np.float32), "radius")

    man = get_manager(bpy.context)
    sim = Simulator()
    man.add(sim)
    item = sim.props
    item.source = source.object
    item.device = "cpu"
    item.gravity = 0.0
    item.use_ground_plane = False
    item.temperature = kT
    item.drag = 10.0
    sim.compile()

    for _ in range(48):  # 2 s at 24 fps, many relaxation times (1 / drag)
        sim.step()
    var = sim.velocity.var(axis=0)
    np.testing.assert_allclose(var, kT / mass, rtol=0.1)

    # Drag alone cools it back down
    item.temperature = 0.0
    for _ in range(24):
        sim.step()
    assert sim.velocity.var() < 0.01 * kT / mass


def test_mesh_edges_become_springs_with_edge_stiffness():
    """GN mesh edges are springs; an EDGE `bond_stiffness` attribute scales each."""
    pos = np.array([[0, 0, 2], [1, 0, 2], [2, 0, 2], [3, 0, 2]], dtype=float)
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    bob = db.BlenderObject.from_mesh(pos, edges=edges, name="Chain")
    stiffness = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    bob.store_named_attribute(stiffness, "bond_stiffness", domain="EDGE")

    man = get_manager(bpy.context)
    sim = Simulator()
    man.add(sim)
    sim.props.source = bob.object
    sim.props.device = "cpu"
    sim.compile()

    assert sim.model.spring_count == 3
    np.testing.assert_allclose(
        sim.model.spring_stiffness.numpy(),  # ty: ignore[unresolved-attribute]
        stiffness * sim.props.spring_ke,
    )
    out = sim.parts[0].output
    assert out is not None and out.object.type == "MESH"
    assert len(out.object.data.edges)  # ty: ignore[unresolved-attribute] == 3


def test_add_protein_network_operator(molecule):
    """The operator sets up and compiles the demo on the active molecule."""
    bpy.context.view_layer.objects.active = molecule  # ty: ignore[unresolved-attribute]
    assert bpy.ops.wb.add_protein_network() == {"FINISHED"}  # ty: ignore[unresolved-attribute]
    item = get_manager(bpy.context).active_item
    assert item.is_compiled
    assert item.source is not None and item.source.name == f"{molecule.name}_network"

    bpy.context.view_layer.objects.active = bpy.data.objects["Camera"]  # ty: ignore[unresolved-attribute]
    assert not bpy.ops.wb.add_protein_network.poll()  # ty: ignore[unresolved-attribute]


@pytest.fixture
def calmodulin() -> bpy.types.Object:
    """AlphaFold model of calmodulin: two lobes joined by a flexible linker."""
    return fetch("P0DP23", database="alphafold")


def test_has_plddt(molecule, calmodulin):
    assert demos.has_plddt(calmodulin)
    assert not demos.has_plddt(molecule)  # 4AKE: crystal B-factors


def test_alphafold_domains_set_contact_stiffness(calmodulin):
    """pLDDT-confident runs become domains; contacts between them are weak."""
    sim = demos.protein_network(bpy.context, calmodulin)
    sim.compile()
    part = sim.parts[0]
    assert part.edges is not None
    domain = part.attributes["domain"][0]
    res_id = part.attributes["res_id"][0]

    # The two lobes, with the central linker and tails left out
    assert set(np.unique(domain)) == {-1, 0, 1}
    lobe1, lobe2 = res_id[domain == 0], res_id[domain == 1]
    assert lobe1.min() <= 10 and 70 <= lobe1.max() < 78
    assert 78 < lobe2.min() <= 86 and lobe2.max() >= 140
    assert {76, 77, 78, 79, 80} <= set(res_id[domain == -1].tolist())

    # Backbone bonds: a chain, pinned at both ends, much stiffer than contacts
    ke = sim.props.spring_ke
    stiffness = sim.model.spring_stiffness.numpy()  # ty: ignore[unresolved-attribute]
    bonds = part.edges[: part.bond_count]
    assert part.bond_count == part.count - 1
    assert (np.abs(np.diff(res_id[bonds], axis=1)) == 1).all()
    np.testing.assert_allclose(stiffness[: part.bond_count], 20 * ke)
    assert sim._pinned_indices is not None
    assert sorted(res_id[sim._pinned_indices].tolist()) == [1, 149]

    # Contacts: ke × √(sᵢsⱼ), weakened unless both beads share a domain ≥ 0
    contacts = part.edges[part.bond_count :]
    k = stiffness[part.bond_count : len(part.edges)]
    d = domain[contacts]
    s_point = part.attributes["stiffness"][0][contacts]
    within = (d[:, 0] == d[:, 1]) & (d[:, 0] >= 0)
    expected = ke * np.sqrt(s_point[:, 0] * s_point[:, 1])
    expected = expected * np.where(within, 1.0, sim.props.domain_coupling)
    np.testing.assert_allclose(k, expected, rtol=1e-5)
    np.testing.assert_allclose(k[within], ke)
    assert within.any() and (~within).any()


def test_pulling_unfolds_and_release_refolds(calmodulin):
    """Pulling the termini breaks contacts; releasing lets them re-form."""
    sim = demos.protein_network(bpy.context, calmodulin)
    sim.props.device = "cpu"
    source = sim.props.source
    assert source is not None
    scene = get_scene()
    scene.frame_set(1)
    sim.compile()
    part = sim.parts[0]
    assert part.edges is not None
    q0 = sim.particle_positions.copy()
    domain = part.attributes["domain"][0]
    bonds = part.edges[: part.bond_count]
    bond_rest = np.linalg.norm(q0[bonds[:, 0]] - q0[bonds[:, 1]], axis=1)

    def lobe_rmsd(q):
        return max(_kabsch_rmsd(q0[domain == k], q[domain == k]) for k in (0, 1))

    def broken():
        return int(sim.broken_springs.sum())

    # Pull to 16 BU over 1 s, hold 0.5 s, release over 0.5 s, rest 3 s (60 fps)
    pull = np.concatenate(
        [
            np.linspace(0, 16, 60),
            np.full(30, 16.0),
            np.linspace(16, 0, 30),
            np.zeros(180),
        ]
    )
    peak_broken, peak_rmsd, max_bond_strain = 0, 0.0, 0.0
    for frame, p in enumerate(pull, start=2):
        demos.set_modifier_input(source, "Pull", float(p))
        scene.frame_set(frame)
        q = sim.particle_positions
        peak_broken = max(peak_broken, broken())
        peak_rmsd = max(peak_rmsd, lobe_rmsd(q))
        strain = np.linalg.norm(q[bonds[:, 0]] - q[bonds[:, 1]], axis=1) / bond_rest - 1
        max_bond_strain = max(max_bond_strain, float(strain.max()))

    assert sim.props.is_active and np.isfinite(sim.particle_positions).all()
    assert peak_broken > 40, "pulling should break native contacts"
    assert peak_rmsd > 0.4, "the lobes should unfold (> 4 Å)"
    assert max_bond_strain < 0.15, "the backbone should hold"
    assert broken() < peak_broken / 3, "contacts should re-form after release"
    final = lobe_rmsd(sim.particle_positions)
    assert final < peak_rmsd / 2, f"the lobes should refold ({final=}, {peak_rmsd=})"


def test_network_edges_carry_strain_and_broken(calmodulin):
    """The output mesh's edges follow spring order, with strain/broken attributes."""
    sim = demos.protein_network(bpy.context, calmodulin)
    sim.props.device = "cpu"
    sim.props.break_strain = 0.01  # break most contacts straight away
    sim.compile()
    for _ in range(5):
        sim.step()
    part = sim.parts[0]
    out = part.output
    assert out is not None and part.edges is not None
    mesh = out.object.data
    edge_verts = np.zeros(len(mesh.edges) * 2, dtype=np.int32)  # ty: ignore[unresolved-attribute]
    mesh.edges.foreach_get("vertices", edge_verts)  # ty: ignore[unresolved-attribute]
    np.testing.assert_array_equal(edge_verts.reshape(-1, 2), part.edges)

    broken = out.named_attribute("broken")
    assert broken.dtype == bool and len(broken) == len(part.edges)
    np.testing.assert_array_equal(
        broken,
        sim.broken_springs[part.spring_start : part.spring_start + len(part.edges)],
    )
    assert not broken[: part.bond_count].any(), "bonds never break"
    assert broken.sum() > 0
    assert len(out.named_attribute("strain")) == len(part.edges)


def test_plddt_smoothing_keeps_p53_dna_binding_domain_whole():
    """p53's DNA-binding domain has a 3-residue pLDDT dip (183-185) that must
    not split it, while its long disordered tails stay soft."""
    p53 = fetch("P04637", database="alphafold")
    sim = demos.protein_network(bpy.context, p53)
    sim.compile()
    attrs = sim.parts[0].attributes
    domain, res_id = attrs["domain"][0], attrs["res_id"][0]
    dbd = domain[(res_id >= 100) & (res_id <= 285)]
    assert len(np.unique(dbd)) == 1 and dbd[0] >= 0
    assert (domain[res_id <= 15] == -1).all()  # disordered N-terminus
    assert (domain[res_id >= 370] == -1).all()  # disordered C-terminus
    stiffness = attrs["stiffness"][0]
    assert (stiffness[domain == -1] < 0.1).all()


def _style_with_atoms_bundle(molecule: bpy.types.Object) -> None:
    """Give the molecule a sphere style whose output carries the atoms in the
    "MN/Atoms" bundle, as Molecular Nodes' Evaluate on Atoms does."""
    from nodebpy import geometry as g

    with g.tree("Bundled Style", clear=True, is_modifier=True) as tree:
        atoms = tree.inputs.geometry("Geometry")
        styled = g.InstanceOnPoints(
            g.MeshToPoints(atoms), instance=g.IcoSphere(radius=0.1)
        )
        bundle = g.StoreBundleItem.geometry(None, "MN/Atoms", atoms)
        styled.o.instances.set_bundle(bundle) >> tree.outputs.geometry("Geometry")
    molecule.modifiers.clear()
    mod = molecule.modifiers.new("MN", "NODES")
    assert isinstance(mod, bpy.types.NodesModifier)
    mod.node_group = tree.tree


@pytest.mark.parametrize("moved", [False, True])
def test_styled_molecule_reads_atoms_from_bundle(calmodulin, moved):
    """Beads come from the "MN/Atoms" bundle, in world space, under any style."""
    _style_with_atoms_bundle(calmodulin)
    if moved:
        calmodulin.location = (1, 2, 3)
        calmodulin.rotation_euler = (0, 0, 0.5)
    sim = demos.protein_network(bpy.context, calmodulin)
    sim.compile()

    ca = db.named_attribute(calmodulin, "is_alpha_carbon")
    local = db.named_attribute(calmodulin, "position")[ca]
    world = np.array(calmodulin.matrix_world)
    expected = local @ world[:3, :3].T + world[:3, 3]
    order = np.argsort(db.named_attribute(calmodulin, "res_id")[ca])
    assert sim.model.particle_count == 149
    np.testing.assert_allclose(sim.particle_positions, expected[order], atol=1e-4)


def test_operator_warns_without_atoms(calmodulin):
    """A style that hides the atoms (no bundle) gives a warning, not a crash."""
    from nodebpy import geometry as g

    with g.tree("Unbundled Style", clear=True, is_modifier=True) as tree:
        atoms = tree.inputs.geometry("Geometry")
        g.InstanceOnPoints(
            g.MeshToPoints(atoms), instance=g.IcoSphere()
        ) >> tree.outputs.geometry("Geometry")
    calmodulin.modifiers.clear()
    mod = calmodulin.modifiers.new("MN", "NODES")
    assert isinstance(mod, bpy.types.NodesModifier)
    mod.node_group = tree.tree

    bpy.context.view_layer.objects.active = calmodulin  # ty: ignore[unresolved-attribute]
    assert bpy.ops.wb.add_protein_network() == {"FINISHED"}  # ty: ignore[unresolved-attribute]
    sim = get_manager(bpy.context).active_simulation
    assert sim.model.particle_count == 0  # ty: ignore[unresolved-attribute]
