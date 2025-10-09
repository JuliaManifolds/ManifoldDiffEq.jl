"""
    ManifoldEuler

The manifold Euler algorithm for problems in the [`ExplicitManifoldODEProblemType`](@ref)
formulation.
"""
struct ManifoldEuler{TM <: AbstractManifold, TR <: AbstractRetractionMethod} <:
    AbstractManifoldDiffEqAlgorithm
    manifold::TM
    retraction_method::TR
end

alg_order(::ManifoldEuler) = 1

"""
    ManifoldEulerCache

Mutable cache for [`ManifoldEuler`](@ref).
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

function perform_step!(integrator, ::ManifoldEulerCache, repeat_step = false)
    u = integrator.u
    t = integrator.t
    alg = integrator.alg

    integrator.k[1] = integrator.f(u, integrator.p, t)
    retract_fused!(
        alg.manifold,
        u,
        u,
        integrator.k[1],
        integrator.dt,
        alg.retraction_method,
    )

    return integrator.stats.nf += 1
end

function initialize!(integrator, ::ManifoldEulerCache)
    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
    integrator.stats.nf += 1
    integrator.kshortsize = 1
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    # Avoid undefined entries if k is an array of arrays
    integrator.fsallast = zero.(integrator.fsalfirst)
    return integrator.k[1] = integrator.fsalfirst
end
