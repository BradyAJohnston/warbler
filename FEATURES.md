# Warbler Feature Roadmap

Interactive GPU simulation in Blender driven by Geometry Nodes.
The core idea: GN sets up geometry and attributes → warbler reads them at compile time (and some per-frame) → Newton simulates on GPU → results write back to Blender objects that GN can read.

---

## GN → Simulation attributes

### Geometry components

| GN output | Simulated as | Output object |
|---|---|---|
| Point cloud | free particles | `<source>_points` (point cloud) |
| Curves | particles + stretch/bend springs along each curve | `<source>_curves` (curves) |
| Mesh with faces | cloth (`add_cloth_mesh`) | `<source>_mesh` (mesh) |
| Mesh without faces | free particles | `<source>_mesh` (point cloud) |
| Instances | one free rigid body each, shaped by `collider` | `<source>_instances` (point cloud + "Warbler Instances" modifier re-placing the source's instances) |

### Currently reading (INSTANCE domain)

| Attribute | Type | Use |
|---|---|---|
| `collider` | `INT` | Collision shape: 0 convex hull (default), 1 box, 2 sphere, 3 capsule, 4 triangle mesh (Set Collider node). Primitives fit the reference's scaled bounding box |
| `velocity` | `FLOAT_VECTOR` | Initial linear velocity |
| `density` | `FLOAT` | Density mass is computed from (default 1000) |

### Currently reading (POINT domain, every component)

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
| `velocity` | `FLOAT_VECTOR` | POINT | Every output object |
| `radius` | `FLOAT` | POINT | Every output object (at compile) |
| `pinned` | `BOOLEAN` | POINT | Every output object (at compile) |
| `is_colliding` | `BOOLEAN` | POINT | Every output object — within 0.1 × radius of a rigid shape / ground |
| `contact_force` | `FLOAT_VECTOR` | POINT | Every output object, VBD only — force from rigid shapes / ground [N] |
| `wb.contact_force` | object prop | — | Rigid objects, VBD only — total force from particles [N] |
| `instance_transform` | `FLOAT4X4` | POINT | Instances output — simulated world transform (scale kept) |
| `velocity` | `FLOAT_VECTOR` | POINT | Instances output — body linear velocity |

### Planned

| Attribute | Type | Domain | Source | Priority |
|---|---|---|---|---|
| `speed` | `FLOAT` | POINT | `‖velocity‖` | low — convenience for GN drivers |
| `contact_normal` | `FLOAT_VECTOR` | POINT | `Contacts` object | low |

---

## New simulation types to add

| Type | Newton API | GN input | Priority |
|---|---|---|---|
| **Granular / MPM** | `SolverImplicitMPM` | Point cloud | medium |
| **Rod cables** | `add_rod` + `SolverVBD` (bending/twist) | Curves | medium — matches Newton's cable examples better than spring chains |

## Rigid body shapes to add

Rigid collection objects: only CUBE is wired up. Source instances already support convex hull / box / sphere / capsule / mesh via `collider`.

| Shape | Newton API | Priority |
|---|---|---|
| Sphere | `add_shape_sphere` | high |
| Capsule | `add_shape_capsule` | high |
| Cylinder | `add_shape_cylinder` | medium |
| Mesh (SDF) | `add_shape_mesh` + `mesh.build_sdf()` | medium |
