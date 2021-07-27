
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
[`ExplicitManifoldODEProblemType`](@ref) formulation. See tableau 6.1 of [^Owren1999].

[^Owren1999]:
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

    k1 = f(u, p, t)
    dt2 = dt / 2
    tmp = retract(alg.manifold, u, k1 * dt2, alg.retraction)
    k2 = f(tmp, p, t + dt2)
    k2t = f.f.operator_vector_transport(tmp, k2, u, p, t)
    retract!(alg.manifold, u, u, dt * k2t, alg.retraction)

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
    interp::InterpolationData,
    retcode = :Default,
    destats = nothing,
    kwargs...,
)
    T = eltype(eltype(u))

    manifold_interp = ManifoldInterpolationData(
        interp.f,
        interp.timeseries,
        interp.ts,
        interp.ks,
        interp.dense,
        interp.cache,
        prob.manifold,
    )
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
        typeof(manifold_interp),
        typeof(destats),
    }(
        u,
        nothing,
        nothing,
        t,
        k,
        prob,
        alg,
        manifold_interp,
        dense,
        0,
        destats,
        retcode,
    )
end
