
"""
    ManifoldLieEuler

The manifold Lie-Euler algorithm for problems in the [`LieODEProblemType`](@ref)
formulation.
"""
struct ManifoldLieEuler{
    TM<:AbstractManifold,
    TR<:AbstractRetractionMethod,
    TG<:AbstractGroupAction,
} <: OrdinaryDiffEqAlgorithm
    manifold::TM
    retraction::TR
    action::TG
end

alg_order(::ManifoldLieEuler) = 1

"""
    ManifoldLieEulerCache

Cache for [`ManifoldEuler`](@ref).
"""
struct ManifoldLieEulerCache{TID<:Identity} <: OrdinaryDiffEqMutableCache
    id::TID
end

"""
    ManifoldLieEulerConstantCache

Cache for [`ManifoldEuler`](@ref).
"""
struct ManifoldLieEulerConstantCache <: OrdinaryDiffEqConstantCache end

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
    return ManifoldLieEulerCache(Identity(base_group(alg.action), f(u, p, t)))
end

@doc raw"""
    apply_diff_group(A::AbstractGroupAction, a, X, p)

For a point on manifold ``p ∈ \mathcal M``` and an element `X` of the tangent space at `a`,
an element of the Lie group of action `A`, ``X ∈ T_a \mathcal G```, compute the
differential of action of `a` on `p` for vector `X`, as specified by rule `A`.
When action on element `p` is written as ``\mathrm{d}τ^p``, with the specified left or right
convention, the differential transforms vectors

````math
(\mathrm{d}τ^p) : T_{a} \mathcal G → T_{τ_a p} \mathcal M
````
"""
apply_diff_group(A::AbstractGroupAction, a, X, p)

function apply_diff_group(
    ::Manifolds.RotationActionOnVector{N,F,LeftAction},
    ::Identity,
    X,
    p,
) where {N,F}
    return X * p
end

function perform_step!(integrator, cache::ManifoldLieEulerCache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p, alg = integrator

    X = f(u, p, t)
    action = alg.action
    k = apply_diff_group(action, cache.id, X, u)

    retract!(alg.manifold, u, u, dt * k, alg.retraction)

    return integrator.destats.nf += 1
end

function initialize!(integrator, cache::ManifoldLieEulerCache)
    @unpack t, uprev, f, p, alg = integrator
    X = f(uprev, p, t) # Pre-start fsal
    integrator.fsalfirst = apply_diff_group(alg.action, cache.id, X, uprev)
    integrator.destats.nf += 1
    integrator.kshortsize = 1
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    # Avoid undefined entries if k is an array of arrays
    integrator.fsallast = zero.(integrator.fsalfirst)
    return integrator.k[1] = integrator.fsalfirst
end
