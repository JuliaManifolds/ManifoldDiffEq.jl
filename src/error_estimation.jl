
"""
    calculate_eest(M::AbstractManifold, utilde, uprev, u, abstol, reltol, internalnorm, t)

Estimate error of a solution of an ODE on manifold `M`.

# Arguments

- `utilde` -- point on `M` for error estimation,
- `uprev` -- point from before the current step,
- `u` -- point after the current step`,
- `abstol` - abolute tolerance,
- `reltol` - relative tolerance,
- `internalnorm` -- copied `internalnorm` from the integrator,
- `t` -- time at which the error is estimated.
"""
function calculate_eest(
    M::AbstractManifold,
    utilde,
    uprev,
    u,
    abstol,
    reltol,
    internalnorm,
    t,
)
    reltol_thing = max(reltol_norm(M, u), reltol_norm(M, uprev))
    return distance(M, u, utilde) / (abstol + reltol_thing * reltol)
end

"""
    reltol_norm(M::AbstractManifold, u)

Estimate the fraction `d_{min}/eps(number_eltype(u))` where `d_{min}` is the distance
between `u`, a point on `M`, and the nearest distinct point on `M` representable in the
representation of `u`.
"""
function reltol_norm(::AbstractManifold, u)
    return norm(u)
end
function reltol_norm(M::ProductManifold, u::ProductRepr)
    mapped_norms = map((Mi, ui) -> reltol_norm(Mi, ui), M.manifolds, u.parts)
    return mean(mapped_norms)
end
