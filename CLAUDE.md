# Warbler — Developer Notes for Claude Code

Blender add-on that bridges Geometry Nodes geometry to GPU-accelerated physics simulation via NVIDIA Newton (built on Warp). Interactive: simulates live in the viewport on frame change.

---

## Architecture

```
GN geometry (points / mesh / curves)
        ↓  GeometrySet  (geometryset.py)
   attributes dict, per component (points / curves / mesh)
        ↓  Simulator.build()  (simulation.py)
   newton.ModelBuilder
        ↓  .finalize()
   newton.Model + State × 2 + SolverXPBD | SolverVBD
        ↓  frame_change_post handler  (manager.py)
   Simulator.step()  every frame
        ↓
   Blender objects updated (rigid body transforms, one output object per source part)
```

### Key files

| File | Purpose |
|---|---|
| `warbler/__init__.py` | `register()` / `unregister()` — wires up classes and the frame handler |
| `warbler/simulation.py` | `Simulator` — the main simulation class; `SourcePart` tracks each component's particle range + output object |
| `warbler/demos.py` | Newton examples rebuilt as nodebpy GN trees + `WB_OT_AddDemo` entries |
| `warbler/nodes/` | Node-group asset sources (MN style) built into the `warbler/assets/nodes.blend` asset library |
| `warbler/geometryset.py` | `GeometrySet` — evaluates GN output, reads attributes |
| `warbler/props.py` | All Blender properties (`SimulationListItem`, `WarblerObjectProperties`) |
| `warbler/manager.py` | `SimulationManager`, `get_manager()`, frame-change handler |
| `warbler/rigid.py` | `RigidObject` wrapper — reads shape config from `wb` properties |
| `warbler/panel.py` | `WB_PT_WarblerPanel` — 3D viewport N-panel |
| `warbler/ops.py` | Operators: Add/Remove/Compile simulation |
| `warbler/utils.py` | `get_scene()`, `wp_transform()`, `blender_rotation()` |
| `FEATURES.md` | Attribute roadmap — what GN attributes map to what Newton params |

---

## Simulation lifecycle

1. User clicks **Compile** → `WB_OT_CompileSimulation` → `Simulator.compile()` → `build()` + `finalize()`
2. `build()` — creates `newton.ModelBuilder`, adds rigid bodies, then `_add_source()` splits the evaluated `props.source` by component:
   - point cloud → free particles
   - curves → particles + springs (`spring_ke` neighbour, `bend_ke` i→i+2)
   - mesh with faces → `add_cloth_mesh` (mesh without faces → free particles)
   - instances → one free rigid body each (`_add_instances`), after the rigid collection's bodies
   Parts are appended in that order, so particle indices are `[points | curves | mesh]`.
3. `finalize()` — `builder.color()` if VBD, `builder.finalize()`, two states, solver (`props.solver`), contacts
4. `_create_outputs()` — one output object per part (`<source>_points` / `_curves` / `_mesh`) at the world origin. On re-compile an existing output (previous compile, or matched by name) with the right object type is **reused**: its data is cleared and refilled in place (`BlenderObject.new_from_pydata` for meshes, `Curves.remove_curves`/`add_curves`, `PointCloud.resize`) so modifiers and materials survive. Outputs whose component disappeared are removed.
5. Every frame change → `_step_simulations(scene, depsgraph)` → `Simulator.step()` (only compiled sims)
6. `step()` order: `_update_simulation_from_blender` (rigid) → `_update_pinned_particles` → `simulate` (substep loop) → `_update_blender_from_simulation` → `_update_outputs`

**Free** (`WB_OT_FreeSimulation` → `Simulator.free()`) drops the model/states/solver and clears `is_compiled`, keeping the outputs, so compile-time settings (solver, device, iterations — locked in the panel while compiled) can be changed. With no live `Simulator` (e.g. after reloading a .blend) it just clears `is_compiled`.

Source positions are local to the source object and moved to world space with `matrix_world` (`_world_props`).

**Solver choice**: XPBD has particle-particle contact but no triangle constraint (cloth stretch comes from edge springs, which is fragile). VBD has proper membrane/bending energies but no particle-particle contact. Particles/ropes → XPBD, cloth → VBD.

---

## GeometrySet priority

`GeometrySet.points` / `.curves` / `.mesh` return `GeometryAttributes` for each component of `eval_obj.evaluated_geometry()` (or the object's own data when there is no GN output). `GeometrySet.data` / `.pointcloud` still return the first of pointcloud → curves → mesh.

**Critical**: data blocks from `evaluated_geometry()` are owned by the temporary GeometrySet and freed when it goes out of scope (`ReferenceError: StructRNA of type Mesh has been removed`). `GeometrySet` keeps `_eval_geom` alive, and `GeometryAttributes` holds a reference to its owning `GeometrySet`, so `GeometrySet(obj).curves.to_props()` is safe.

---

## Newton patterns

- `particle_grid` is `None` unless `particle_count > 1` AND `particle_max_radius > 0`. Never call `particle_grid.build()` — the solver does it internally.
- Collision: create `newton.CollisionPipeline(model)` once in `finalize()`, pre-allocate `contacts = pipeline.contacts()`, then per substep `pipeline.collide(state, contacts)` and `solver.step(state_in, state_out, control, contacts, dt)`. (`model.contacts()` / `model.collide()` are deprecated since newton 1.5.)
- `newton.use_coord_layout_targets = True` is set at import in `simulation.py` — silences the legacy joint_target_q layout DeprecationWarning; warbler never writes joint targets. A checkout of newton's dev branch lives at `../newton` (examples + `docs/migration.rst`).
- `particle_q_init` is cloned from `state_in.particle_q` at the start of every `solver.step()`. Writing updated positions for kinematic (pinned) particles into `state_0.particle_q` before the step automatically propagates those positions as the kinematic target — no need to touch `particle_q_init` directly.
- Rigid box sizes are live: `model.shape_scale` (box half-extents) and `model.shape_collision_radius` (broadphase sphere, `‖half-extents‖` for a box) are read by the collision pipeline every collide, so `_update_rigid_scales` writes the objects' world-space size (`RigidObject.half_extents()`: local `bound_box` × `matrix_world` scale) without a re-compile. The change is ramped over the frame's substeps (`_apply_rigid_scales`) — applied in one substep it buries particles and XPBD launches them. Mass/inertia are not updated.
- Rigid bodies use **world** transforms: read via `utils.world_transform()` (`matrix_world.decompose()`), written back for active bodies by assigning `matrix_world` (Blender solves the local transform, so parents and any rotation mode work; scale is preserved). `build()` calls `view_layer.update()` first because `matrix_world` is stale after a script edits `location` etc. Kinematic bodies get linear **and** angular velocity in `body_qd` (linear first, angular world-frame) from frame-to-frame differences.
- `add_body(is_kinematic=True)` for user-controlled rigid bodies. Kinematic bodies collide with particles (velocity-driven impulses) but are never moved by physics forces. Previously these were fighting the per-frame position override.
- **Contact outputs** (newton ≥ 1.7): `kernels.gather_soft_contacts` runs after each frame's last substep over the particle/shape contact records → POINT `is_colliding` (gap ≤ 0.1·radius) on every output. VBD also reports forces via the experimental observables API (`solver.observables({CONTACT_F}, contacts=...)`, passed to `step()` on the last substep only; soft rows sit after `rigid_contact_max` and are the force *on the body*) → POINT `contact_force` and the rigid object's `wb.contact_force`. XPBD's `contact_f` only has body-body rows, so it gets `is_colliding` alone. Overhead is unmeasurable.
- **Attachments**: `builder.add_attachment_particle_body` (VBD only; XPBD ignores them — warbler warns). Anchor the particles *outside* the body: one that starts inside also collides with it and fights the spring (the carried-cloth bar sits one radius clear of the edge).
- **Cloth self-collision** (`cloth_self_contact`, default on; VBD with triangles only): `particle_enable_self_contact` with `particle_self_contact_margin` (= "Thickness", 0.02) and `particle_self_contact_gap` (0.01) — `example_cloth_hanging`'s values. Works with the external pipeline and graph capture; roughly doubles cloth cost (5 → 10 ms hanging cloth, 7 → 19 ms when piled). Piled cloth keeps layers ≥ margin apart (0.6 cm without).
- **VBD options**: `rigid_compliant_alm=True` always (omitting it is deprecated). `rigid_soft_log_barrier` / `rigid_soft_dat` map to `rigid_soft_contact_use_log_barrier` / `rigid_soft_enable_dat`. DAT requires a solver-owned pipeline (`SolverVBD(collision_pipeline=...)`), so then `self.contacts = solver.contacts`, warbler skips `collide()` and passes `contacts=None` to `step()` (`_solver_collides`); ~35% slower.
- **CUDA graphs** (`use_cuda_graph`, default on): `simulate()` runs the frame's substep loop (`_simulate_frame`) normally once, then records it with `wp.ScopedCapture` and replays it with `wp.capture_launch`. Slow cloth was Python launch overhead, not GPU work — VBD launches hundreds of kernels per frame (~10 µs each): cloth 23 → 4 ms/frame, carried cloth 46 → 7, ropes 8 → 3. Rules for code in the substep loop: no host reads/writes or numpy (per-substep inputs live on the device — `_noise_seed` advanced by `kernels.advance_seed`, box scale ramp via `kernels.lerp_shape_scale` from `_scale_start/_target`), no allocation, and write per-frame inputs (pins, kinematic bodies) *in place* into `state_0`. Kernel scalars are baked in, so they go in `_graph_settings()`'s key (re-records on change). Odd substep counts end with `state_1.assign(state_0)` + swap so every frame starts on the same buffers. Not recorded: CPU, a `surface` (mesh id changes every frame). A solver-owned pipeline (DAT) records fine (73 → 15 ms). Replays are bit-identical to plain launches (XPBD springs vary ~1e-4 run to run either way, from atomics). `step()` synchronizes after `simulate()` so `time_compute` is real.
- **Instances → rigid bodies** (`_add_instances`): `GeometrySet.instances` wraps `evaluated_geometry().instances_pointcloud()` (POINT `instance_transform` FLOAT4X4 — databpy returns it transposed, translation in row 3 — and `.reference_index`) and `instance_references()` (GeometrySets, or Objects/Collections; only meshes are used, an instance without one gets no body: `InstancePart.body == -1`). Body = instance's world loc/rot; scale goes into the shape. INSTANCE `collider` indexes `COLLIDERS` (0 hull, 1 box, 2 sphere, 3 capsule, 4 mesh); one `newton.Mesh` per reference, hulls rebuilt as a fresh `Mesh` so inertia comes from the closed hull (Suzanne is open). Body indices: rigid collection objects first, so `_update_simulation_from_blender` rewrites only those rows. Instance shapes use the sim's `instance_ke/kd/mu` (1e8 / 5e5 / 1.0, `example_basic_shapes`' VBD values): newton's default `ke` 2.5e3 lets VBD bodies sink through the ground (VBD averages the two shapes' ke). Output `<source>_instances` is a point cloud of world `instance_transform`s; its "Warbler Instances" modifier takes the source's instances via Object Info and `Set Instance Transform`s them by index, so references, attributes and materials come from the source. A hidden source still evaluates. Primitive colliders are fitted boxes that turn with the instance — start them apart or they're thrown out of the overlap; spheres/capsules never settle (no rolling resistance).
- **Particles ↔ rigid bodies** (curves/points vs instances in one source): XPBD couples them both ways (a body pushing a strand loses the momentum the strand gains; `test_curves_and_instances_push_each_other`) and also has particle–particle contact, so strands collide with each other. VBD doesn't: a strand is an immovable wall to a body (body stops dead, strand doesn't move). Mixed scenes → XPBD.
- **Interactions** (non-collision attraction): binding sites are vertices of an *instanced mesh* with POINT `binding_type` (int) and/or `binding` (strength) — read by `InstanceAttributes.reference_geometry()` → `ReferenceMesh` — stored body-local × instance scale in `_sites`; particles opt in with POINT `affinity_type` / `affinity` (`_interaction_types`: a strength alone means type 1, a type alone strength 1). Which types attract is the **interaction table**, kept in the source: points flagged POINT `interaction` with `interaction_a/_b/_depth/_range` (the Interaction node group makes one row). `GeometrySet.points` drops those rows from the particles (`GeometryAttributes._rows` mask, also hiding them from outputs) and `GeometrySet.interactions` returns them. A row works both ways; unlisted pairs don't interact; no table at all → every pair uses the sim's `binding_depth/width` (the untyped fallback). The (types × types) `depth` / `width` device arrays are sized at compile and refilled **in place every frame** (`_update_interactions`), so rows can be animated without re-compiling or re-recording the graph (`test_interaction_table_is_live`). Kernels (each substep after the thermostat), both Gaussian wells `-depth·strengths·exp(-r²/2w²)` cut at 3w via `kernels._well`: `binding_forces` (particle ↔ site; one thread per typed particle, bodies beyond reach + 3·max width skipped; POINT `bound` within 2w) and `site_site_forces` (body ↔ body sites, 2D launch over pairs of site-carrying bodies, upper triangle; equal and opposite wrenches; `_body_partners` → INSTANCE-output POINT `partners`). Equal and opposite (`test_binding_conserves_momentum`). Depth is in kT units: particle ↔ site 0.5 at kT 0.05 sticks but unbinds; patch ↔ patch meets over many site pairs at once, so a much smaller per-pair depth (crowd: 0.05) — and slow-diffusing proteins need a wider range (0.1) and low density (5) to find each other within a shot. Region-specific = sites on a patch; non-specific = all vertices. Cost ~1 ms per 1.7k particle-facing sites; the crowd with dimers ~9 ms/frame. Node groups: Set Binding Sites, Set Affinity (both with Type), Interaction.
- **Binding wells must not press into colliders** (the "pairs gain energy and fly off" bug): Newton's particle–shape contacts (XPBD) keep the world-space contact normal from `collide()` while the contact point rotates with the body, so a *sustained* push gives a torque ∝ ω — angular momentum isn't conserved and a bound body spins itself up exponentially (friction only slows it; frictionless it doubled every ~50 frames). Transient contacts never show it; a binding well buried under the collider (a vertex in a hull dimple, or a site the bead can't reach) holds beads against the surface for good. Fix: `collider_surface` moves every site onto the shape the body collides with (nearest hull facet from the cached hull mesh, box face, sphere/capsule surface; MESH keeps vertex + normal), and `binding_forces` centres each particle's well one particle radius out along the site normal, where it rests. Site ↔ site uses the surface sites directly (touching surfaces = well minimum). Guarded by `test_binding_does_not_spin_bodies_up` (fails with the wells pulled half a radius in) and angular-momentum checks; wall-less crowd hottest protein went 106 kT → 7.6 kT. Worth reporting upstream.
- **Curve rest shape**: bend springs (i → i+2) take their rest length from the starting curve, so a curve that starts coiled springs back to the coil. `bend_rest_straight` rests them straight instead (rest = the two segments end to end) — needed when something else (binding) holds a curve bent, e.g. DNA on a histone. i → i+2 springs resist gentle bends only weakly (length change is second order in the angle).
- **Site ↔ site is skipped** unless the interaction table has a row pairing two *site* types (`_site_pairs`, decided at compile; depth stays live). Dense sites (nucleosome "Slide": 115 per histone) made the unused all-pairs loop cost 10 ms/frame.
- **Warp kernels vs ruff**: ruff's UP018 turns `int(0)` into `0`, but a variable mutated inside a dynamic loop in a Warp kernel must be declared with a type constructor — use `wp.int32(0)` / `wp.float32(0.0)`, which ruff leaves alone. `ty check --fix` (pre-commit) deletes `# ty: ignore`s that ruff's reformat moved off their line; re-add them on the new line.
- **Bead spacing on chains in XPBD**: particle contact acts between *neighbouring* beads of a strand too. At ≤ 2r spacing it pushes them apart against their bond springs — bonds stretch to 2r and the strand buckles even at zero temperature. Space beads ≥ 3r (`ResampleCurve(mode="Length", length=3·radius)`); the half-bead gaps are still too narrow for another strand to pass through.
- `builder.particle_mass` is a plain Python list before `finalize()` — you can zero out pinned particles by index: `builder.particle_mass[i] = 0.0`.

---

## Node groups (`warbler/nodes/`)

Same pipeline as Molecular Nodes. `warbler/nodes/geometry/` holds one node-group asset per module (`AssetGeometryGroup`, `ASSET` + `ASSET_METADATA` catalog id), the source of truth for the asset library `warbler/assets/nodes.blend` (gitignored, built):

```bash
uv run nodebpy build   # sources → warbler/assets/nodes.blend + catalogs ([tool.nodebpy.assets] in pyproject)
uv run nodebpy dump    # .blend → sources: regenerates docstrings, typed __init__, accessors from _build_group
uv run nodebpy check   # build → dump reproduces the sources byte-for-byte
```

Edit a group's `_build_group` (and `_tree_properties["description"]`), then `build` + `dump` so the typed boilerplate follows; `#` comments inside `_build_group` don't survive a dump — explain in socket/tree `description=`s, or (MN's `mn-nodes` skill convention) with a `with g.Frame("Title"):` block holding an unlinked `g.String("prose")` next to the nodes it explains (create those nodes inside the block, not inline in a call outside it, or they aren't in the frame); dump renames the orphan `_string`. Per-point field inputs get `hide_value=True`. `uv run nodebpy plot warbler/assets/nodes.blend <outdir> "<Name>"` renders a tree to PNG for review. A new group: write a module with `_name`/`_asset_name`/`_library`/`_build_group`/`ASSET`/`ASSET_METADATA` (catalog UUID from `warbler/nodes/blender_assets.cats.txt`), then build + dump (which also rewrites `geometry/__init__.py`). Every non-`_` `.py` under `warbler/nodes/` must be an asset module — runtime helpers are `_`-prefixed (`_library.py` registers the "Warbler" asset library in preferences at addon register, so the groups show in Add > Warbler/<catalog> and the Asset Browser). At runtime the classes link the group from the `.blend` (packed); without a built `.blend` (fresh checkout) they build from `_build_group` — tests pass either way; `test_asset_library` skips without it. `release.yml` runs `nodebpy build` before `extbpy build` (`required_files` makes extbpy refuse to package without it).

Catalogs: Warbler/Particles (Set Particles, Set Pinned, Set Attached, Set On Surface, Set Velocity — POINT and INSTANCE), Warbler/Rigid Bodies (Set Collider — menu → INSTANCE `collider`, order must match `simulation.COLLIDERS`; Set Density), Warbler/Generators (Point Grid, Cloth Grid with an `Edge` field output), Warbler/Interactions (Set Binding Sites — POINT `binding_type`/`binding` on a mesh to be instanced; Set Affinity — POINT `affinity_type`/`affinity`; Interaction — one interaction-table row, joined into the source), Warbler/Outputs (Simulated Instances — the instances output's modifier wraps it in the "Warbler Instances" tree, whose Geometry input must come first). Demo trees use these (`from .nodes import geometry as nodes`).

## GN attribute conventions

Warbler reads these named attributes from geometry at compile time:

**POINT domain (particles)**

| Attribute | Type | Used for |
|---|---|---|
| `position` | `FLOAT_VECTOR` | Initial positions (required) |
| `velocity` | `FLOAT_VECTOR` | Initial velocities |
| `mass` | `FLOAT` | Per-particle mass |
| `radius` | `FLOAT` | Collision radius |
| `pinned` | `BOOLEAN` | `True` → kinematic (sets `ParticleFlags` to 0, position driven by GN each frame) |
| `on_surface` | `BOOLEAN` | Held on the simulation's `surface` mesh (all unpinned particles if absent and a surface is set) |
| `surface_offset` | `FLOAT` | Distance along the surface normal to hold the particle at (e.g. ± half a bilayer) |
| `attach` | `BOOLEAN` | VBD only: held by a particle-body spring (`attach_stiffness` / `attach_damping`) to the nearest rigid body, anchored where it starts |

These are read on the POINT domain of every component (points, curve points, mesh vertices). `pinned=True` sets mass 0 and `ParticleFlags` 0. Pin positions are re-read from GN every frame (for all parts) so animated pins work — see the ropes demo.

---

## Testing

```bash
BLENDER_USER_EXTENSIONS=/nonexistent uv run pytest tests/   # all tests (env var stops an installed extension's nodebpy shadowing the venv's)
uv run pre-commit run --all-files  # ruff format + ruff check + ty check
```

Tests use `bpy` (the Python package, not a running Blender). The `clean_and_save` fixture in `conftest.py` loads a fresh homefile before each test and saves a `.blend` to `tests/blend_files/` after for inspection.

Test files:
- `test_simulation.py` — rigid body compile + dynamics
- `test_geometryset.py` — GN attribute reading (mesh, pointcloud, vertex-only)
- `test_demos.py` — each demo checked against its Newton example's `test_final` criteria; mixed-component source; demo operator

Demos (`warbler/demos.py`) reproduce Newton examples with the geometry built in GN via nodebpy: particle grid (`add_particle_grid` from `example_xpbd_vbd_coupled_solver`), hanging cloth (`example_cloth_hanging`, VBD), hanging ropes (spring chains with GN-animated pins — Newton's own cables use `add_rod` rigid bodies instead), carried cloth (VBD cloth whose edge is `attach`ed to a keyframed kinematic bar, dragged over an obstacle box; output material coloured by `is_colliding`), falling monkeys (16×16×10 randomly turned Suzanne instances as XPBD convex-hull bodies onto the ground; 2560 bodies ≈ 7 ms/frame compute, 2 s compile), molecular crowd (64 lumpy protein instances as convex-hull bodies + 16 DNA curves in a box of 6 kinematic walls, zero gravity, temperature 0.05 / drag 5, XPBD, `bend_ke` 1e3; DNA starts straight in the channels between the protein grid's layers — strands started through proteins get kinked as contacts push their beads out; output curves get a "Warbler Tube" modifier sized by `radius`; proteins have a "Binding Patch" (cap around +Z, area fraction; 1 = everywhere): two of the three blob shapes are DNA binders (type 1, green patch), one a dimeriser (type 2, yellow patch on purple); DNA is type 3 (red where `bound`); table rows (1, 3) "DNA Binding" and (2, 2) "Dimer Binding"/"Dimer Range" are modifier inputs; protein density 5 so they diffuse; ~9 ms/frame), nucleosomes (one DNA curve wound 1.5 left-handed turns round each of 6 capsule histones, in and out on the same side; units alternate 180° about X so each unit's entry meets the previous exit — a zig-zag fibre; built as ordered points per unit → `PointsToCurves(weight=unit·N + index)` → resampled at 3r. Histone = capsule mesh + loose groove vertices (Set Binding Sites type 1) projected onto the capsule; DNA type 2; "Slide" = sites every r/2 (smooth track: the wrap slides along the groove when pulled) vs one per bead (each bead holds its well, the histone rolls instead — `test_nucleosome_slide_or_stick` measures drift in the histone frame, since a free histone is a pulley and feeds DNA out either way); `bend_rest_straight` so only binding holds the wrap; "Pin Ends"/"Pull" move the last bead. Newton's own cable/pulley example (`example_cable_cross_slide_table`) is rods on VBD held on by tension + friction — no tension in a bath, so attraction (binding) is what holds DNA on; 5–6 ms/frame). Demos whose output re-places the source's instances hide the source in viewport *and* renders (`_hide_source`). They set scene fps to 60 to match Newton's timing.

**Protein elastic network** (`demos.protein_network`, operator `wb.add_protein_network` on the active MN molecule, `tests/test_protein.py`): GN picks chain-A Cα atoms via MN's `is_alpha_carbon` attribute (not MN's Python classes — MN as an extension isn't importable as `molecularnodes`); springs come from `network_cutoff` (`simulation.network_edges`, compile-time, exact). Mesh edges always become permanent bond springs (EDGE `bond_stiffness`/`bond_damping` multipliers — not `stiffness`, which is the POINT contact multiplier; a mesh can't hold both). All source POINT attributes are copied to outputs, so MN styles can run on the simulated mesh. Thermostat: `kernels.langevin_forces` (`temperature` = kT, `drag` = 1/s) added to `particle_f` each substep; `kernels.langevin_body_forces` does the same for dynamic rigid bodies into `body_f` (world wrench about the COM): force from mass, torque from the body-frame inertia (`-drag·I·ω` + `√I` noise via `wp.eig3`), so ⟨m·v²⟩ = kT per axis and ⟨ω·I·ω⟩ = 3kT in both solvers. Defaults at MN's 0.1 BU/Å: cutoff 0.8, ke 1e3, kT 10, drag 5 → ~1 Å RMSD jiggle. MN is a dev-only dep, pinned to a git commit until 520.3.0 is on PyPI. Test structures are downloaded with `mn.Molecule.fetch` into `tests/.cache/` (gitignored) on first run — never commit structure files.

Domains & unfolding: the protein tree joins consecutive Cα by mesh edges (bonds, `bond_stiffness` 20× — weaker bonds tear before contacts break), and with "Domains from pLDDT" (auto-on via `demos.has_plddt`: AlphaFold writes one value per residue to all its atoms) smooths `b_factor ≥ 70` over ±3 residues, numbering confident runs as `domain` (else -1, `stiffness` 0.05). Contact k = `spring_ke·√(sᵢsⱼ)`, × `domain_coupling` unless same domain ≥ 0. `kernels.update_network_contacts` runs every substep: contact stiffness → 0 above `break_strain` (XPBD skips ke ≤ 0), restored below `reform_strain`. Output mesh edges are in spring order with EDGE `strain`/`broken`. "Pull" modifier input moves pinned termini apart live. Test models: 4AKE (RCSB), AlphaFold P0DP23 (calmodulin) and P04637 (p53), via `fetch(..., database="alphafold")` in `test_protein.py` (MN goes through biotite; AFDB 403s plain curl).

**Membrane fusion** (`demos.membrane_fusion`): the `surface` prop is a mesh object re-read and rebuilt as a `wp.Mesh` every frame (`_update_surface`; rebuilt, not refit, since remeshes change topology). `kernels.project_to_surface` runs after every substep: closest point + interpolated vertex normal × `surface_offset`, normal velocity removed; outputs get `surface_normal`. The demo surface is a smooth-min SDF of two spheres in a Volume Cube → Volume to Mesh (background must be negative, else the cube walls become surface), radius grown to keep area. The lipid source scatters on the surface via Object Info but holds the result in a pass-through simulation zone, so it's scattered once (re-scattering the animated surface every frame cost ~40 ms). Settings from the user's tuned example: lipid radius 0.044 with Poisson distance min = radius (starts over-packed, contacts spread it into a fluid — sparse packing clumped instead), temperature 0.02, drag 5 → ~23% of lipids cross vesicles by frame 400.

MN's `BuildElasticNetwork` group (MN main @ 9059f03) outputs points up to 1.5 Å off the real Cα positions with mismatched `res_id`s — don't use it as a source until fixed.

Blender 5.2: GN modifier inputs are no longer ID properties — use `getattr(mod.properties.inputs, socket.identifier).value`.

---

## Running in Blender

Extension metadata lives in `[tool.extbpy]` in `pyproject.toml`; [extbpy](https://github.com/BradyAJohnston/extbpy) merges it with `[project]` and `uv.lock` to generate `warbler/blender_manifest.toml` and `warbler/wheels/` (both gitignored — never edit by hand).

```bash
uvx extbpy sync      # write manifest + wheels into warbler/ for loading from source
uvx extbpy manifest  # preview the generated manifest
uvx extbpy build     # per-platform zips (also syncs); validates with `blender` if on PATH
```

Re-run `sync` after changing dependencies. For development, load the `warbler/` folder directly as an extension via Blender's developer mode. `.github/workflows/ci.yml` (push/PR, mirrors MN's): ruff, `nodebpy check` roundtrip, and `extbpy build --blender blender` — extbpy runs `blender --command extension validate` on every zip — uploaded as a 7-day artifact (installable test build per commit). Validate locally with `uvx extbpy build --blender <path to Blender 5.2>` (without `--blender`/Blender on PATH the check is silently skipped). Tagging `v*` runs `.github/workflows/release.yml`: version set from the tag (`uv version --no-sync`), `nodebpy build`, the same validated `extbpy build` into `dist/`, then a **published** (not draft) `--latest` release whose assets are the zips renamed without the version (`warbler-<platform>.zip`) plus an `index.json` from `blender --command extension server-generate` — so `https://github.com/BradyAJohnston/warbler/releases/latest/download/index.json` is a self-updating extension repo. The README's drag-and-drop buttons (`docs/img/install/*.svg`) link to `releases/latest/download/warbler-<platform>.zip?repository=./index.json&blender_version_min=5.2.0&platforms=<platform>`: Blender (`bl_pkg` `_invoke_for_drop`) resolves `./index.json` against the zip URL, offers to add that repo, and installs only if the stripped dropped URL equals an index `archive_url` resolved against the repo URL — hence the stable names. The extensions aggregator (BradyAJohnston/extensions, `repo/source.txt`) also indexes the latest published release daily — drafts/pre-releases are invisible to it. An optional `EXTENSIONS_DISPATCH_TOKEN` secret triggers its rebuild immediately (needs a `repository_dispatch` trigger there). Releases made with `GITHUB_TOKEN` don't trigger other workflows, so the docs deploy from pushes to main. README's preview is `docs/img/interactive_curves.gif` (GitHub strips `<video>`/mp4 from repo paths).

The blender debug session starts with `-Xfrozen_modules=off` passed to Python to make the debugger hit breakpoints.

---

## Known issues / watch-outs

- `source_evaluate` property exists in props but is not yet used — currently always evaluates
- nodebpy's signatures don't accept tuples mixing numbers and sockets (`(0, 0, height)`), though they work at runtime; `demos.py` uses `# ty: ignore[invalid-argument-type]` on those lines
- `rigid_decay_frames` prop still exists but `smooth_lerp` was removed when kinematic body handling was fixed. The prop now has no effect.
- `spring_kf` is unused (`spring_ke` / `spring_kd` / `bend_ke` drive curve springs).

---

## Next steps (from FEATURES.md)

**High priority:**
1. Additional rigid body shapes for the rigid collection — sphere, capsule, cylinder, mesh (source instances already have them via `collider`)
2. Simulation write-back — store `contact_force` / `is_colliding` as named attributes on output objects so GN can react to collisions

**Medium priority:**
3. Granular / MPM — `SolverImplicitMPM` for sand-like behaviour
4. Two-way coupling — rigid bodies reacting to particles

**Cleanup:**
- `source_evaluate` should control whether GN is re-evaluated at particle read time
