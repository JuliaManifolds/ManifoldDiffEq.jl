
"""
ManifoldLieEuler

The Lie-Euler algorithm for problems in the [`ExplicitManifoldODEProblemType`](@ref)
formulation.
"""
struct ManifoldLieEuler{TM<:AbstractManifold,TR<:AbstractRetractionMethod} <:
       OrdinaryDiffEqAlgorithm
    manifold::TM
    retraction::TR
end

alg_order(::ManifoldLieEuler) = 1

"""
LieEulerCache

Cache for [`ManifoldLieEuler`](@ref).
"""
struct LieEulerCache <: OrdinaryDiffEqMutableCache end

"""
LieEulerConstantCache

Cache for [`ManifoldLieEuler`](@ref).
"""
struct LieEulerConstantCache <: OrdinaryDiffEqConstantCache end

function alg_cache(
    alg::ManifoldLieEuler,
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
    return LieEulerCache()
end

function perform_step!(integrator, cache::LieEulerCache, repeat_step = false)
    @unpack t, dt, uprev, u, p, alg = integrator

    L = integrator.f.f

    update_coefficients!(L, u, p, t)

    retract!(alg.manifold, u, u, dt * L(u, p, t), alg.retraction)

    integrator.f(integrator.fsallast, u, p, t + dt)
    return integrator.destats.nf += 1
end

function initialize!(integrator, cache::LieEulerCache)
    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
    integrator.destats.nf += 1
    integrator.kshortsize = 1
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    # Avoid undefined entries if k is an array of arrays
    integrator.fsallast = zero.(integrator.fsalfirst)
    return integrator.k[1] = integrator.fsalfirst
end

function build_solution(
    prob::ManifoldODEProblem,
    alg,
    t,
    u;
    timeseries_errors = length(u) > 2,
    dense = false,
    dense_errors = dense,
    calculate_error = true,
    k = nothing,
    interp = LinearInterpolation(t, u),
    retcode = :Default,
    destats = nothing,
    kwargs...,
)

    T = eltype(eltype(u))
    f = prob.f

    return ODESolution{
        T,
        1,
        typeof(u),
        Nothing,
        Nothing,
        typeof(t),
        typeof(k),
        typeof(prob),
        typeof(alg),
        typeof(interp),
        typeof(destats),
    }(
        u,
        nothing,
        nothing,
        t,
        k,
        prob,
        alg,
        interp,
        dense,
        0,
        destats,
        retcode,
    )
end
