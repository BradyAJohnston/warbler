"""Warp kernels warbler adds on top of Newton's solvers."""

import warp as wp

PARTICLE_ACTIVE = wp.constant(1)  # newton.ParticleFlags.ACTIVE
BODY_KINEMATIC = wp.constant(2)  # newton.BodyFlags.KINEMATIC
# Offsets the bodies' random streams from the particles' (same seed, same tid)
BODY_STREAM = wp.constant(1 << 30)


@wp.kernel
def langevin_forces(
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    drag: float,
    kT: float,
    dt: float,
    seed: wp.array[wp.int32],
    substep: int,
    particle_f: wp.array[wp.vec3],
):
    """Add Langevin drag and random thermal kicks to each active particle.

    ``f = -drag·m·v + sqrt(2·drag·m·kT / dt)·N(0, 1)``, which holds each
    velocity component at variance ``kT / m``. The solvers integrate
    ``particle_f``, so this runs after ``clear_forces()`` each substep.
    The seed lives on the device (``advance_seed``) so a captured CUDA graph
    still draws fresh noise every frame.
    """
    tid = wp.tid()
    m = particle_mass[tid]
    if (particle_flags[tid] & PARTICLE_ACTIVE) == 0 or m <= 0.0:
        return
    rng = wp.rand_init(seed[0] + substep, tid)  # ty: ignore[invalid-argument-type]
    noise = wp.vec3(wp.randn(rng), wp.randn(rng), wp.randn(rng))
    kick = wp.sqrt(2.0 * drag * m * kT / dt)
    particle_f[tid] = particle_f[tid] - drag * m * particle_qd[tid] + kick * noise  # ty: ignore[invalid-assignment]


@wp.kernel
def langevin_body_forces(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],  # ty: ignore[invalid-type-form]
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_flags: wp.array[wp.int32],
    drag: float,
    kT: float,
    dt: float,
    seed: wp.array[wp.int32],
    substep: int,
    body_f: wp.array[wp.spatial_vector],  # ty: ignore[invalid-type-form]
):
    """``langevin_forces`` for rigid bodies: linear and angular.

    The force uses the mass, the torque the inertia tensor: in the body frame
    ``τ = -drag·I·ω + sqrt(2·drag·kT / dt)·√I·N(0, 1)`` with ``√I`` from its
    eigen-decomposition, which holds each rotational mode at ``kT``. Newton's
    ``body_f`` is a world-frame wrench about the centre of mass.
    """
    tid = wp.tid()
    m = body_mass[tid]
    if (body_flags[tid] & BODY_KINEMATIC) != 0 or m <= 0.0:
        return
    rng = wp.rand_init(seed[0] + substep, tid + BODY_STREAM)
    qd = body_qd[tid]
    v = wp.spatial_top(qd)
    w = wp.spatial_bottom(qd)
    kick = wp.sqrt(2.0 * drag * kT / dt)

    noise = wp.vec3(wp.randn(rng), wp.randn(rng), wp.randn(rng))
    f = -drag * m * v + kick * wp.sqrt(m) * noise

    rot = wp.transform_get_rotation(body_q[tid])
    inertia = body_inertia[tid]
    axes, moments = wp.eig3(inertia)
    root = wp.diag(
        wp.vec3(
            wp.sqrt(wp.max(moments[0], 0.0)),  # ty: ignore[not-subscriptable]
            wp.sqrt(wp.max(moments[1], 0.0)),  # ty: ignore[not-subscriptable]
            wp.sqrt(wp.max(moments[2], 0.0)),  # ty: ignore[not-subscriptable]
        )
    )
    noise = wp.vec3(wp.randn(rng), wp.randn(rng), wp.randn(rng))
    w_body = wp.quat_rotate_inv(rot, w)
    sqrt_inertia = axes * root * wp.transpose(axes)  # ty: ignore[unsupported-operator]
    tau_body = -drag * (inertia * w_body) + kick * (sqrt_inertia * noise)
    tau = wp.quat_rotate(rot, tau_body)

    body_f[tid] = body_f[tid] + wp.spatial_vector(f, tau)  # ty: ignore[invalid-assignment]


@wp.kernel
def advance_seed(seed: wp.array[wp.int32], count: int):
    seed[0] = seed[0] + count  # ty: ignore[invalid-assignment]


@wp.kernel
def lerp_shape_scale(
    shapes: wp.array[wp.int32],
    start: wp.array[wp.vec3],
    target: wp.array[wp.vec3],
    t: float,
    shape_scale: wp.array[wp.vec3],
    shape_collision_radius: wp.array[float],
):
    """Set box half-extents ``t`` of the way from ``start`` to ``target``,
    with the broadphase radius (``‖half-extents‖``) to match."""
    tid = wp.tid()
    scale = start[tid] + (target[tid] - start[tid]) * t
    shape_scale[shapes[tid]] = scale  # ty: ignore[invalid-assignment]
    shape_collision_radius[shapes[tid]] = wp.length(scale)  # ty: ignore[invalid-assignment]


@wp.kernel
def update_network_contacts(
    contact_ids: wp.array[wp.int32],
    spring_indices: wp.array[wp.int32],
    spring_rest_length: wp.array[float],
    native_stiffness: wp.array[float],
    particle_q: wp.array[wp.vec3],
    break_strain: float,
    reform_strain: float,
    broken: wp.array[wp.int32],
    spring_stiffness: wp.array[float],
):
    """Break network contacts stretched past ``break_strain``; re-form them
    once back under ``reform_strain`` (a Gō-like native-contact rule).

    A broken contact has zero stiffness, which the solvers skip. With
    ``break_strain`` ≤ 0 nothing breaks and broken contacts re-form.
    """
    tid = wp.tid()
    s = contact_ids[tid]
    i = spring_indices[2 * s]
    j = spring_indices[2 * s + 1]
    strain = wp.length(particle_q[i] - particle_q[j]) / spring_rest_length[s] - 1.0
    if broken[tid] == 0:
        if break_strain > 0.0 and strain > break_strain:
            broken[tid] = 1  # ty: ignore[invalid-assignment]
            spring_stiffness[s] = 0.0  # ty: ignore[invalid-assignment]
    elif strain < reform_strain or break_strain <= 0.0:
        broken[tid] = 0  # ty: ignore[invalid-assignment]
        spring_stiffness[s] = native_stiffness[tid]  # ty: ignore[invalid-assignment]


@wp.kernel
def project_to_surface(
    mesh: wp.uint64,
    vertex_normals: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    offset: wp.array[float],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
):
    """Snap particles onto a surface mesh and keep only tangential velocity.

    Each particle moves to the closest point on the mesh, plus ``offset``
    along the smoothly interpolated vertex normal (e.g. ± half a bilayer
    thickness for the two leaflets).
    """
    tid = wp.tid()
    i = indices[tid]
    query = wp.mesh_query_point_no_sign(mesh, particle_q[i], 1.0e6)
    if not query.result:  # ty: ignore[unresolved-attribute]
        return
    face = query.face  # ty: ignore[unresolved-attribute]
    u = query.u  # ty: ignore[unresolved-attribute]
    v = query.v  # ty: ignore[unresolved-attribute]
    a = wp.mesh_get_index(mesh, face * 3 + 0)
    b = wp.mesh_get_index(mesh, face * 3 + 1)
    c = wp.mesh_get_index(mesh, face * 3 + 2)
    n = wp.normalize(
        vertex_normals[a] * u
        + vertex_normals[b] * v
        + vertex_normals[c] * (1.0 - u - v)
    )
    normal[tid] = n  # ty: ignore[invalid-assignment]
    point = wp.mesh_eval_position(mesh, face, u, v)
    particle_q[i] = point + n * offset[tid]  # ty: ignore[invalid-assignment]
    qd = particle_qd[i]
    particle_qd[i] = qd - wp.dot(qd, n) * n  # ty: ignore[invalid-assignment]


@wp.kernel
def gather_soft_contacts(
    soft_contact_count: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[int],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    contact_f: wp.array[wp.spatial_vector],  # ty: ignore[invalid-type-form]
    rigid_contact_max: int,
    tolerance: float,
    particle_contact_f: wp.array[wp.vec3],
    particle_colliding: wp.array[wp.int32],
    body_contact_f: wp.array[wp.vec3],
):
    """Per-particle contact state from the particle/rigid-shape contact records.

    A record's particles are colliding when the contact point is within
    ``tolerance`` × radius of the shape surface. ``contact_f`` holds the
    solver's reported wrench per record (rigid rows first, then soft rows,
    the force on the shape's body); pass an empty array when the solver
    doesn't report soft contacts. The force on the particles is the negation,
    split by the record's barycentric weights.
    """
    tid = wp.tid()
    if tid >= soft_contact_count[0]:
        return
    idx = soft_contact_indices[tid]
    bary = soft_contact_barycentric[tid]
    body = shape_body[soft_contact_shape[tid]]
    X_wb = wp.transform_identity()
    if body >= 0:
        X_wb = body_q[body]
    surface = wp.transform_point(X_wb, soft_contact_body_pos[tid])  # ty: ignore[no-matching-overload]

    x = wp.vec3(0.0)
    r = 0.0
    for k in range(3):
        if idx[k] >= 0:
            x += bary[k] * particle_q[idx[k]]
            r += bary[k] * particle_radius[idx[k]]
    gap = wp.dot(soft_contact_normal[tid], x - surface) - r

    f = wp.vec3(0.0)
    if contact_f.shape[0] > 0:
        f = wp.spatial_top(contact_f[rigid_contact_max + tid])  # ty: ignore[unsupported-operator]
        if body >= 0:
            wp.atomic_add(body_contact_f, body, f)  # ty: ignore[invalid-argument-type]

    for k in range(3):
        p = idx[k]
        if p >= 0 and bary[k] > 0.0:
            wp.atomic_add(particle_contact_f, p, -f * bary[k])  # ty: ignore[invalid-argument-type, unsupported-operator]
            if gap <= tolerance * r:
                particle_colliding[p] = 1  # ty: ignore[invalid-assignment]
