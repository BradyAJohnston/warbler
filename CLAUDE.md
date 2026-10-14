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
- `builder.particle_mass` is a plain Python list before `finalize()` — you can zero out pinned particles by index: `builder.particle_mass[i] = 0.0`.

---

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

Demos (`warbler/demos.py`) reproduce Newton examples with the geometry built in GN via nodebpy: particle grid (`add_particle_grid` from `example_xpbd_vbd_coupled_solver`), hanging cloth (`example_cloth_hanging`, VBD), hanging ropes (spring chains with GN-animated pins — Newton's own cables use `add_rod` rigid bodies instead). They set scene fps to 60 to match Newton's timing.

**Protein elastic network** (`demos.protein_network`, operator `wb.add_protein_network` on the active MN molecule, `tests/test_protein.py`): GN picks chain-A Cα atoms via MN's `is_alpha_carbon` attribute (not MN's Python classes — MN as an extension isn't importable as `molecularnodes`); springs come from `network_cutoff` (`simulation.network_edges`, compile-time, exact). Mesh edges always become permanent bond springs (EDGE `bond_stiffness`/`bond_damping` multipliers — not `stiffness`, which is the POINT contact multiplier; a mesh can't hold both). All source POINT attributes are copied to outputs, so MN styles can run on the simulated mesh. Thermostat: `kernels.langevin_forces` (`temperature` = kT, `drag` = 1/s) added to `particle_f` each substep. Defaults at MN's 0.1 BU/Å: cutoff 0.8, ke 1e3, kT 10, drag 5 → ~1 Å RMSD jiggle. MN is a dev-only dep, pinned to a git commit until 520.3.0 is on PyPI. Test structures are downloaded with `mn.Molecule.fetch` into `tests/.cache/` (gitignored) on first run — never commit structure files.

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

Re-run `sync` after changing dependencies. For development, load the `warbler/` folder directly as an extension via Blender's developer mode. Tagging `v*` runs `.github/workflows/release.yml` (`extbpy build` → draft release).

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
1. Additional rigid body shapes — sphere, capsule, cylinder (`add_shape_sphere`, `add_shape_capsule`), and mesh SDF (`add_shape_mesh` + `mesh.build_sdf()`)
2. Simulation write-back — store `contact_force` / `is_colliding` as named attributes on output objects so GN can react to collisions

**Medium priority:**
3. Granular / MPM — `SolverImplicitMPM` for sand-like behaviour
4. Two-way coupling — rigid bodies reacting to particles

**Cleanup:**
- `source_evaluate` should control whether GN is re-evaluated at particle read time
