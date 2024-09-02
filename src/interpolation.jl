

"""
    struct ManifoldInterpolationData end


Inspired by `OrdinaryDiffEq.OrdinaryDiffEqInterpolation`.
"""
struct ManifoldInterpolationData{F,uType,tType,kType,cacheType,TM}
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

function manifold_linear_interpolation(
    M::AbstractManifold,
    Θ,
    dt,
    p,
    q,
    idxs::Nothing,
    T::Type{Val{0}},
)
    return shortest_geodesic(M, p, q, Θ)
end

function manifold_linear_interpolation(
    M::AbstractManifold,
    Θ,
    dt,
    ps,
    qs,
    idxs,
    T::Type{Val{0}},
)
    return map(i -> shortest_geodesic(M, ps[i], qs[i], Θ), idxs)
end

function ode_interpolation(
    tval::Number,
    id::ManifoldInterpolationData,
    idxs,
    deriv,
    p,
    continuity::Symbol = :left,
)
    # implmenented based on `ode_interpolation` from OrdinaryDiffEq.jl
    @unpack ts, timeseries, ks, f, cache = id
    @inbounds tdir = sign(ts[end] - ts[1])

    if continuity === :left
        # we have i₋ = i₊ = 1 if tval = ts[1], i₊ = i₋ + 1 = lastindex(ts) if tval > ts[end],
        # and otherwise i₋ and i₊ satisfy ts[i₋] < tval ≤ ts[i₊]
        i₊ = min(lastindex(ts), OrdinaryDiffEq._searchsortedfirst(ts, tval, 2, tdir > 0))
        i₋ = i₊ > 1 ? i₊ - 1 : i₊
    else
        # we have i₋ = i₊ - 1 = 1 if tval < ts[1], i₊ = i₋ = lastindex(ts) if tval = ts[end],
        # and otherwise i₋ and i₊ satisfy ts[i₋] ≤ tval < ts[i₊]
        i₋ = max(1, OrdinaryDiffEq._searchsortedlast(ts, tval, 1, tdir > 0))
        i₊ = i₋ < lastindex(ts) ? i₋ + 1 : i₋
    end

    begin
        dt = ts[i₊] - ts[i₋]
        Θ = iszero(dt) ? oneunit(tval) / oneunit(dt) : (tval - ts[i₋]) / dt

        # TODO: use manifold Hermite interpolation
        val = manifold_linear_interpolation(
            id.manifold,
            Θ,
            dt,
            timeseries[i₋],
            timeseries[i₊],
            idxs,
            deriv,
        )
    end

    return val
end
