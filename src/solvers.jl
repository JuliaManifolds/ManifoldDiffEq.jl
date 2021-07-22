

struct ManifoldInterpolationData{F,uType,tType,kType,cacheType,TM} <: OrdinaryDiffEq.OrdinaryDiffEqInterpolation{cacheType}
    f::F
    timeseries::uType
    ts::tType
    ks::kType
    dense::Bool
    cache::cacheType
    manifold::TM
end

function (interp::ManifoldInterpolationData)(t, idxs, deriv, p, continuity)
    return ode_interpolation(t, interp, idxs, deriv, p, continuity)
end

function manifold_linear_interpolation(M::AbstractManifold, Θ, dt, p, q, idxs::Nothing,T::Type{Val{0}})
    return shortest_geodesic(M, p, q, Θ)
end

function manifold_linear_interpolation(M::AbstractManifold, Θ, dt, ps, qs, idxs, T::Type{Val{0}})
    return map(i -> shortest_geodesic(M, ps[i], qs[i], Θ), idxs)
end

function ode_interpolation(tval::Number, id::ManifoldInterpolationData, idxs, deriv, p, continuity::Symbol=:left)
    # implmenented based on `ode_interpolation` from OrdinaryDiffEq.jl
    @unpack ts,timeseries,ks,f,cache = id
    @inbounds tdir = sign(ts[end]-ts[1])

    if continuity === :left
        # we have i₋ = i₊ = 1 if tval = ts[1], i₊ = i₋ + 1 = lastindex(ts) if tval > ts[end],
        # and otherwise i₋ and i₊ satisfy ts[i₋] < tval ≤ ts[i₊]
        i₊ = min(lastindex(ts), OrdinaryDiffEq._searchsortedfirst(ts,tval,2,tdir > 0))
        i₋ = i₊ > 1 ? i₊ - 1 : i₊
    else
        # we have i₋ = i₊ - 1 = 1 if tval < ts[1], i₊ = i₋ = lastindex(ts) if tval = ts[end],
        # and otherwise i₋ and i₊ satisfy ts[i₋] ≤ tval < ts[i₊]
        i₋ = max(1, OrdinaryDiffEq._searchsortedlast(ts,tval,1,tdir > 0))
        i₊ = i₋ < lastindex(ts) ? i₋ + 1 : i₋
    end
    
    begin
        dt = ts[i₊] - ts[i₋]
        Θ = iszero(dt) ? oneunit(tval) / oneunit(dt) : (tval-ts[i₋]) / dt

        # TODO: use manifold Hermite interpolation
        val = manifold_linear_interpolation(id.manifold,Θ,dt,timeseries[i₋],timeseries[i₊],idxs,deriv)
    end

    return val
end



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
    interp::InterpolationData,
    retcode = :Default,
    destats = nothing,
    kwargs...,
)
    T = eltype(eltype(u))
    
    manifold_interp = ManifoldInterpolationData(interp.f, interp.timeseries, interp.ts, interp.ks, interp.dense, interp.cache, prob.manifold)
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
