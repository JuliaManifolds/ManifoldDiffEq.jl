
"""
    ManifoldEuler

The manifold Euler algorithm for problems in the [`ExplicitManifoldODEProblemType`](@ref)
formulation.
"""
struct ManifoldEuler{TM<:AbstractManifold,TR<:AbstractRetractionMethod} <:
       OrdinaryDiffEqAlgorithm
    manifold::TM
    retraction::TR
end

alg_order(::ManifoldEuler) = 1

"""
    ManifoldEulerCache

Cache for [`ManifoldEuler`](@ref).
"""
struct ManifoldEulerCache <: OrdinaryDiffEqMutableCache end

"""
    ManifoldEulerConstantCache

Cache for [`ManifoldEuler`](@ref).
"""
struct ManifoldEulerConstantCache <: OrdinaryDiffEqConstantCache end

function alg_cache(
    alg::ManifoldEuler,
    u,
    rate_prototype,
    uEltypeNoUnits,
    uBottomEltypeNoUnits,
    tTypeNoUnits,
    uprev,
    uprev2,
    f,
    t,
    dt,
    reltol,
    p,
    calck,
    ::Val{true},
)
    return ManifoldEulerCache()
end

function perform_step!(integrator, cache::ManifoldEulerCache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p, alg = integrator

    k = f(u, p, t)
    retract!(alg.manifold, u, u, dt * k, alg.retraction)

    return integrator.destats.nf += 1
end

function initialize!(integrator, cache::ManifoldEulerCache)
    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
    integrator.destats.nf += 1
    integrator.kshortsize = 1
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    # Avoid undefined entries if k is an array of arrays
    integrator.fsallast = zero.(integrator.fsalfirst)
    return integrator.k[1] = integrator.fsalfirst
end


"""
    CG2

A Crouch-Grossmann algorithm of second order for problems in the
[`ExplicitManifoldODEProblemType`](@ref) formulation. See order 2 conditions discussed
in [^OwrenMarthinsen1999]. Tableau:

0    | 0
1/2  | 1/2  0
----------------
     | 0    1

[^OwrenMarthinsen1999]:
    > B. Owren and A. Marthinsen, “Runge-Kutta Methods Adapted to Manifolds and Based on
    > Rigid Frames,” BIT Numerical Mathematics, vol. 39, no. 1, pp. 116–142, Mar. 1999,
    > doi: 10.1023/A:1022325426017.

"""
struct CG2{TM<:AbstractManifold,TR<:AbstractRetractionMethod} <: OrdinaryDiffEqAlgorithm
    manifold::TM
    retraction::TR
end

alg_order(::CG2) = 2

"""
    CG2Cache

Cache for [`CG2`](@ref).
"""
struct CG2Cache <: OrdinaryDiffEqMutableCache end


function alg_cache(
    alg::CG2,
    u,
    rate_prototype,
    uEltypeNoUnits,
    uBottomEltypeNoUnits,
    tTypeNoUnits,
    uprev,
    uprev2,
    f,
    t,
    dt,
    reltol,
    p,
    calck,
    ::Val{true},
)
    return CG2Cache()
end

function perform_step!(integrator, cache::CG2Cache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p, alg = integrator

    M = alg.manifold
    k1 = f(u, p, t)
    dt2 = dt / 2
    tmp = retract(M, u, k1 * dt2, alg.retraction)
    k2 = f(tmp, p, t + dt2)
    k2t = f.f.operator_vector_transport(M, tmp, k2, u, p, t + dt2, t)
    retract!(M, u, u, dt * k2t, alg.retraction)

    return integrator.destats.nf += 2
end

function initialize!(integrator, cache::CG2Cache)
    integrator.kshortsize = 2
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t)
    integrator.destats.nf += 1

    integrator.fsallast = zero.(integrator.fsalfirst)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    return nothing
end


"""
    CG3

A Crouch-Grossmann algorithm of second order for problems in the
[`ExplicitManifoldODEProblemType`](@ref) formulation. See tableau 6.1 of [^OwrenMarthinsen1999]:

 0     | 0
 3/4   | 3/4      0
 17/24 | 119/216  17/108  0
 ------------------------------
       | 13/51    -2/3    24/17

[^OwrenMarthinsen1999]:
    > B. Owren and A. Marthinsen, “Runge-Kutta Methods Adapted to Manifolds and Based on
    > Rigid Frames,” BIT Numerical Mathematics, vol. 39, no. 1, pp. 116–142, Mar. 1999,
    > doi: 10.1023/A:1022325426017.

"""
struct CG3{TM<:AbstractManifold,TR<:AbstractRetractionMethod} <: OrdinaryDiffEqAlgorithm
    manifold::TM
    retraction::TR
end

alg_order(::CG3) = 3

"""
    CG3Cache

Cache for [`CG3`](@ref).
"""
struct CG3Cache <: OrdinaryDiffEqMutableCache end


function alg_cache(
    alg::CG3,
    u,
    rate_prototype,
    uEltypeNoUnits,
    uBottomEltypeNoUnits,
    tTypeNoUnits,
    uprev,
    uprev2,
    f,
    t,
    dt,
    reltol,
    p,
    calck,
    ::Val{true},
)
    return CG3Cache()
end

function perform_step!(integrator, cache::CG3Cache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p, alg = integrator
    M = alg.manifold

    k1 = f(u, p, t)
    c2h = (3 // 4) * dt
    c3h = (17 // 24) * dt
    a21h = (3 // 4) * dt
    a31h = (119 // 216) * dt
    a32h = (17 // 108) * dt
    b1 = (13 // 51) * dt
    b2 = (-2 // 3) * dt
    b3 = (24 // 17) * dt
    k2u = retract(M, u, k1 * a21h, alg.retraction)
    k2 = f(k2u, p, t + c2h)
    k1tk2u = f.f.operator_vector_transport(M, u, k1, k2u, p, t, t + c2h)
    k3u = retract(M, k2u, a31h * k1tk2u + a32h * k2)
    k3 = f(k3u, p, t + c3h)

    k2tu = f.f.operator_vector_transport(M, k2u, k2, u, p, t + c2h, t)
    k3tu = f.f.operator_vector_transport(M, k3u, k3, u, p, t + c3h, t)
    retract!(M, u, u, b1 * k1 + b2 * k2tu + b3 * k3tu, alg.retraction)

    return integrator.destats.nf += 2
end

function initialize!(integrator, cache::CG3Cache)
    integrator.kshortsize = 2
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t)
    integrator.destats.nf += 1

    integrator.fsallast = zero.(integrator.fsalfirst)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    return nothing
end
