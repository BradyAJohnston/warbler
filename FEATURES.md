# Warbler Feature Roadmap

Interactive GPU simulation in Blender driven by Geometry Nodes.
The core idea: GN sets up geometry and attributes → warbler reads them at compile time (and some per-frame) → Newton simulates on GPU → results write back to Blender objects that GN can read.

---

## GN → Simulation attributes

### Currently reading (POINT domain)

| Attribute | Type | Domain | Use |
|---|---|---|---|
| `position` | `FLOAT_VECTOR` | POINT | Initial particle positions |
| `velocity` | `FLOAT_VECTOR` | POINT | Initial particle velocities |
| `mass` | `FLOAT` | POINT | Per-particle mass |
| `radius` | `FLOAT` | POINT | Collision radius |

---

### Implemented

#### `pinned` — BOOL, POINT domain
Marks a point as kinematically fixed. Pinned particles are not moved by physics forces; their positions are driven by the GN tree instead.

- `True` → particle flags = 0 (kinematic, zero effective mass)
- `False` / absent → particle flags = `ACTIVE` (physics-driven)

**Dynamic pins**: pinned particle positions are re-read from the evaluated GN tree every frame, before the solve. This means animated or GN-driven pin positions propagate into the simulation automatically — move a pinned point in GN and the particles respond.

---

### Planned — POINT domain

| Attribute | Type | Priority | Notes |
|---|---|---|---|
| `color_group` | `INT` | medium | Collision group filtering. Negative = collide with everything except same group. Maps to `ShapeConfig.collision_group` |

Material properties below are **global** in Newton's XPBD (not per-particle). Read as a constant scalar from GN to drive the simulation panel values, or expose in a future SolverVBD path.

| Attribute | Type | Priority | Notes |
|---|---|---|---|
| `ke` | `FLOAT` | low | Contact stiffness → `model.soft_contact_ke` |
| `kd` | `FLOAT` | low | Contact damping → `model.soft_contact_kd` |
| `mu` | `FLOAT` | low | Friction → `model.soft_contact_mu` |

---

## Simulation → GN write-back

### Currently writing

| Attribute | Type | Domain | Object |
|---|---|---|---|
| `position` | `FLOAT_VECTOR` | POINT | Particle point cloud |
| `velocity` | `FLOAT_VECTOR` | POINT | Particle point cloud |

### Planned

| Attribute | Type | Domain | Source | Priority |
|---|---|---|---|---|
| `speed` | `FLOAT` | POINT | `‖velocity‖` | low — convenience for GN drivers |
| `contact_force` | `FLOAT_VECTOR` | POINT | `state_0.particle_f` after step | medium |
| `is_colliding` | `BOOLEAN` | POINT | `Contacts` object | medium |
| `contact_normal` | `FLOAT_VECTOR` | POINT | `Contacts` object | low |

---

## New simulation types to add

| Type | Newton API | GN input | Priority |
|---|---|---|---|
| **Granular / MPM** | `SolverImplicitMPM` | Point cloud | medium |

## Rigid body shapes to add

Currently only CUBE is wired up.

| Shape | Newton API | Priority |
|---|---|---|
| Sphere | `add_shape_sphere` | high |
| Capsule | `add_shape_capsule` | high |
| Cylinder | `add_shape_cylinder` | medium |
| Mesh (SDF) | `add_shape_mesh` + `mesh.build_sdf()` | medium |
