"""Warp kernels warbler adds on top of Newton's solvers."""

import warp as wp

PARTICLE_ACTIVE = wp.constant(1)  # newton.ParticleFlags.ACTIVE


@wp.kernel
def langevin_forces(
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    drag: float,
    kT: float,
    dt: float,
    seed: int,
    particle_f: wp.array[wp.vec3],
):
    """Add Langevin drag and random thermal kicks to each active particle.

    ``f = -drag·m·v + sqrt(2·drag·m·kT / dt)·N(0, 1)``, which holds each
    velocity component at variance ``kT / m``. The solvers integrate
    ``particle_f``, so this runs after ``clear_forces()`` each substep.
    """
    tid = wp.tid()
    m = particle_mass[tid]
    if (particle_flags[tid] & PARTICLE_ACTIVE) == 0 or m <= 0.0:
        return
    rng = wp.rand_init(seed, tid)  # ty: ignore[invalid-argument-type]
    noise = wp.vec3(wp.randn(rng), wp.randn(rng), wp.randn(rng))
    kick = wp.sqrt(2.0 * drag * m * kT / dt)
    particle_f[tid] = particle_f[tid] - drag * m * particle_qd[tid] + kick * noise  # ty: ignore[invalid-assignment]


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
