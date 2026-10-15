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


@wp.func
def _well(d: wp.vec3, strength: float, depth: float, width: float) -> wp.vec3:
    """Force on the end of ``d`` (= x - site) from a Gaussian well
    ``U = -strength·depth·exp(-r²/2w²)`` at the site: toward it, 0 beyond 3w."""
    r2 = wp.dot(d, d)
    if r2 > 9.0 * width * width:
        return wp.vec3(0.0)
    inv_w2 = 1.0 / (width * width)
    return -(strength * depth * wp.exp(-0.5 * r2 * inv_w2) * inv_w2) * d


@wp.kernel
def binding_forces(
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_type: wp.array[wp.int32],
    particle_affinity: wp.array[float],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    site_bodies: wp.array[wp.int32],
    body_site_start: wp.array[wp.int32],
    body_site_end: wp.array[wp.int32],
    body_reach: wp.array[float],
    site_local: wp.array[wp.vec3],
    site_normal: wp.array[wp.vec3],
    site_type: wp.array[wp.int32],
    site_strength: wp.array[float],
    pair_depth: wp.array2d[float],
    pair_width: wp.array2d[float],
    max_width: wp.array[float],
    particle_f: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],  # ty: ignore[invalid-type-form]
    particle_bound: wp.array[wp.int32],
):
    """Attraction between typed particles and typed binding sites on bodies.

    Sites lie on the body's collision surface (``collider_surface``), and each
    particle's well is centred one particle radius out along the site normal,
    where the particle rests against the surface: bound particles sit in the
    well instead of being pressed into the collider.

    The pair table (``pair_depth`` / ``pair_width``, indexed by the two
    interaction types) gives each site a Gaussian well for each particle type
    (see ``_well``), scaled by the site's strength and the particle's
    affinity; a depth of 0 means the pair doesn't interact. The body gets the
    opposite force at the site, a wrench about its centre of mass. Bodies are
    skipped when the particle is beyond their sites' reach plus the widest
    well. ``particle_bound`` marks particles within 2 widths of a site they
    interact with.
    """
    tid = wp.tid()
    a = particle_type[tid]
    if a <= 0:
        return
    affinity = particle_affinity[tid]
    x = particle_q[tid]
    radius = particle_radius[tid]
    reach = 3.0 * max_width[0] + radius
    f_particle = wp.vec3(0.0)
    bound = wp.int32(0)  # dynamic: a literal 0 is constant in a Warp loop
    for k in range(site_bodies.shape[0]):
        b = site_bodies[k]
        xform = body_q[b]
        origin = wp.transform_get_translation(xform)
        if wp.length(x - origin) > body_reach[k] + reach:
            continue
        com = wp.transform_point(xform, body_com[b])
        f_body = wp.vec3(0.0)
        tau_body = wp.vec3(0.0)
        for s in range(body_site_start[k], body_site_end[k]):
            depth = pair_depth[site_type[s], a]  # ty: ignore[not-subscriptable]
            if depth <= 0.0:
                continue
            width = pair_width[site_type[s], a]  # ty: ignore[not-subscriptable]
            rest = site_local[s] + radius * site_normal[s]
            site = wp.transform_point(xform, rest)
            d = x - site
            if wp.dot(d, d) < 4.0 * width * width:
                bound = wp.int32(1)
            f = _well(d, affinity * site_strength[s], depth, width)
            f_particle += f
            f_body -= f
            tau_body -= wp.cross(site - com, f)  # ty: ignore[unsupported-operator]
        if wp.length(f_body) > 0.0:
            wp.atomic_add(body_f, b, wp.spatial_vector(f_body, tau_body))  # ty: ignore[invalid-argument-type]
    particle_f[tid] = particle_f[tid] + f_particle  # ty: ignore[invalid-assignment]
    particle_bound[tid] = bound  # ty: ignore[invalid-assignment]


@wp.kernel
def site_site_forces(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    site_bodies: wp.array[wp.int32],
    body_site_start: wp.array[wp.int32],
    body_site_end: wp.array[wp.int32],
    body_reach: wp.array[float],
    site_local: wp.array[wp.vec3],
    site_type: wp.array[wp.int32],
    site_strength: wp.array[float],
    pair_depth: wp.array2d[float],
    pair_width: wp.array2d[float],
    max_width: wp.array[float],
    body_f: wp.array[wp.spatial_vector],  # ty: ignore[invalid-type-form]
    body_partners: wp.array[wp.int32],
):
    """Attraction between the binding sites of two bodies (e.g. protein
    interfaces), from the same pair table as ``binding_forces``.

    One thread per pair of bodies with sites (a 2D launch; the lower triangle
    returns at once); pairs beyond their reaches plus the widest well are
    skipped. Each site pair is a Gaussian well, scaled by both strengths,
    pulling the sites together: equal and opposite wrenches on the bodies.
    ``body_partners`` counts, per body, the bodies it is bound to (a site
    pair within 2 widths).
    """
    i, j = wp.tid()  # ty: ignore[invalid-assignment, not-iterable]
    if j <= i:
        return
    b1 = site_bodies[i]
    b2 = site_bodies[j]
    x1 = body_q[b1]
    x2 = body_q[b2]
    gap = wp.length(wp.transform_get_translation(x1) - wp.transform_get_translation(x2))  # ty: ignore[unsupported-operator]
    if gap > body_reach[i] + body_reach[j] + 3.0 * max_width[0]:
        return
    com1 = wp.transform_point(x1, body_com[b1])
    com2 = wp.transform_point(x2, body_com[b2])
    f1 = wp.vec3(0.0)
    tau1 = wp.vec3(0.0)
    tau2 = wp.vec3(0.0)
    bound = wp.int32(0)  # dynamic: a literal 0 is constant in a Warp loop
    for s1 in range(body_site_start[i], body_site_end[i]):
        p1 = wp.transform_point(x1, site_local[s1])
        for s2 in range(body_site_start[j], body_site_end[j]):
            depth = pair_depth[site_type[s1], site_type[s2]]  # ty: ignore[not-subscriptable]
            if depth <= 0.0:
                continue
            width = pair_width[site_type[s1], site_type[s2]]  # ty: ignore[not-subscriptable]
            p2 = wp.transform_point(x2, site_local[s2])
            d = p1 - p2  # ty: ignore[unsupported-operator]
            if wp.dot(d, d) < 4.0 * width * width:
                bound = wp.int32(1)
            f = _well(d, site_strength[s1] * site_strength[s2], depth, width)
            f1 += f
            tau1 += wp.cross(p1 - com1, f)  # ty: ignore[unsupported-operator]
            tau2 -= wp.cross(p2 - com2, f)  # ty: ignore[unsupported-operator]
    if wp.length(f1) > 0.0:
        wp.atomic_add(body_f, b1, wp.spatial_vector(f1, tau1))  # ty: ignore[invalid-argument-type]
        wp.atomic_add(body_f, b2, wp.spatial_vector(-f1, tau2))  # ty: ignore[invalid-argument-type]
    if bound:
        wp.atomic_add(body_partners, b1, 1)  # ty: ignore[invalid-argument-type]
        wp.atomic_add(body_partners, b2, 1)  # ty: ignore[invalid-argument-type]


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
