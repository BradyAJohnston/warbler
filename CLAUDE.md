# Warbler — Developer Notes for Claude Code

Blender add-on that bridges Geometry Nodes geometry to GPU-accelerated physics simulation via NVIDIA Newton (built on Warp). Interactive: simulates live in the viewport on frame change.

---

## Architecture

```
GN geometry (points / mesh / curves)
        ↓  GeometrySet  (geometryset.py)
   attributes dict
        ↓  SimulatorXPBD.build()  (simulation.py)
   newton.ModelBuilder
        ↓  .finalize()
   newton.Model + State × 2 + SolverXPBD
        ↓  frame_change_post handler  (manager.py)
   SimulatorXPBD.step()  every frame
        ↓
   Blender objects updated (rigid body transforms, particle pointcloud)
```

### Key files

| File | Purpose |
|---|---|
| `warbler/__init__.py` | `register()` / `unregister()` — wires up classes and the frame handler |
| `warbler/simulation.py` | `SimulatorXPBD` — the main simulation class |
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

1. User clicks **Compile** → `WB_OT_CompileSimulation` → `SimulatorXPBD.compile()` → `build()` + `finalize()`
2. `build()` — creates `newton.ModelBuilder`, adds rigid bodies, particles
3. `finalize()` — calls `builder.finalize()`, creates two states, solver, contacts; creates Blender output objects (`ParticleObject`)
4. Every frame change → `_step_simulations(scene, depsgraph)` → `SimulatorXPBD.step()`
5. `step()` order: `_update_simulation_from_blender` → `simulate` (substep loop) → `_update_blender_from_simulation` → visualisation write-backs

---

## GeometrySet priority

`GeometrySet.data` checks `eval_obj.evaluated_geometry()` in order:
**pointcloud → curves → mesh** (GN output), then falls back to `eval_obj.data` (base object).

**Critical**: `_eval_geom` must be stored as an instance attribute. Data blocks from `evaluated_geometry()` are owned by the temporary GeometrySet and freed when it goes out of scope. Any access after losing the reference causes `ReferenceError: StructRNA of type Mesh has been removed`.

---

## Newton patterns

- `particle_grid` is `None` unless `particle_count > 1` AND `particle_max_radius > 0`. Never call `particle_grid.build()` — the solver does it internally.
- Collision: create `newton.CollisionPipeline(model)` once in `finalize()`, pre-allocate `contacts = pipeline.contacts()`, then per substep `pipeline.collide(state, contacts)` and `solver.step(state_in, state_out, control, contacts, dt)`. (`model.contacts()` / `model.collide()` are deprecated since newton 1.5.)
- `newton.use_coord_layout_targets = True` is set at import in `simulation.py` — silences the legacy joint_target_q layout DeprecationWarning; warbler never writes joint targets. A checkout of newton's dev branch lives at `../newton` (examples + `docs/migration.rst`).
- `particle_q_init` is cloned from `state_in.particle_q` at the start of every `solver.step()`. Writing updated positions for kinematic (pinned) particles into `state_0.particle_q` before the step automatically propagates those positions as the kinematic target — no need to touch `particle_q_init` directly.
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

For particles, `pinned=True` sets `ParticleFlags` to 0 (not ACTIVE). Pin positions are re-read from GN every frame so animated pins work.

---

## Testing

```bash
uv run pytest tests/          # all tests
uv run pytest tests/ -v       # verbose
uv run pre-commit run --all-files  # ruff format + ruff check + ty check
```

Tests use `bpy` (the Python package, not a running Blender). The `clean_and_save` fixture in `conftest.py` loads a fresh homefile before each test and saves a `.blend` to `tests/blend_files/` after for inspection.

Test files:
- `test_simulation.py` — rigid body compile + dynamics
- `test_geometryset.py` — GN attribute reading (mesh, pointcloud, vertex-only)

---

## Running in Blender

The add-on is loaded as an extension from `warbler/blender_manifest.toml`. Wheels are bundled under `warbler/wheels/`. For development, load the folder directly as an extension via Blender's developer mode.

The blender debug session starts with `-Xfrozen_modules=off` passed to Python to make the debugger hit breakpoints.

---

## Known issues / watch-outs

- `particle_source_evaluate` property exists in props but is not yet used — currently always evaluates
- `rigid_decay_frames` prop still exists but `smooth_lerp` was removed when kinematic body handling was fixed. The prop now has no effect.
- Spring properties (`spring_ke`, `spring_kd`, `spring_kf`) are stored on `SimulationListItem` but `_add_springs()` is empty — not yet wired up.
- The `rigid_decay_frames` UI prop in the Rigid Bodies panel no longer does anything.

---

## Next steps (from FEATURES.md)

**High priority:**
1. Additional rigid body shapes — sphere, capsule, cylinder (`add_shape_sphere`, `add_shape_capsule`), and mesh SDF (`add_shape_mesh` + `mesh.build_sdf()`)
2. Simulation write-back — store `contact_force` / `is_colliding` as named attributes on output objects so GN can react to collisions

**Medium priority:**
3. Granular / MPM — `SolverImplicitMPM` for sand-like behaviour
4. Two-way coupling — rigid bodies reacting to particles

**Cleanup:**
- Remove `rigid_decay_frames` from the panel (or re-implement as something useful)
- Remove `spring_ke/kd/kf` from the panel until springs are actually implemented
- `particle_source_evaluate` should control whether GN is re-evaluated at particle read time
