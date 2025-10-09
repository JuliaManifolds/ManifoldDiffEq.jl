
"""
    ManifoldLieEuler

The manifold Lie-Euler algorithm for problems in the [`LieODEProblemType`](@ref)
formulation.
"""
struct ManifoldLieEuler{
    TR<:AbstractRetractionMethod,
    TA<:GroupAction,
} <: AbstractManifoldDiffEqAlgorithm
    retraction_method::TR
    action::TA
end

SciMLBase.alg_order(::ManifoldLieEuler) = 1

"""
    ManifoldLieEulerCache

Mutable cache for [`ManifoldLieEuler`](@ref).
"""
struct ManifoldLieEulerCache{TID<:LieGroups.Identity} <: OrdinaryDiffEqMutableCache
    id::TID
end

"""
    ManifoldLieEulerConstantCache

Constant cache for [`ManifoldLieEuler`](@ref).
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
    return ManifoldLieEulerCache(Identity(base_lie_group(alg.action)))
end

function perform_step!(integrator, cache::ManifoldLieEulerCache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p, alg = integrator
    X = f(u, p, t)
    k = diff_group_apply(alg.action, cache.id, X, u)
    retract_fused!(alg.manifold, u, u, k, dt, alg.retraction_method)
    return integrator.stats.nf += 1
end

function initialize!(integrator, cache::ManifoldLieEulerCache)
    @unpack t, uprev, f, p, alg = integrator
    X = f(uprev, p, t) # Pre-start fsal
    integrator.fsalfirst =
        copy(base_manifold(alg.action), diff_group_apply(alg.action, cache.id, X, uprev))
    integrator.stats.nf += 1
    integrator.kshortsize = 1
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)
    # Avoid undefined entries if k is an array of arrays
    integrator.fsallast = zero.(integrator.fsalfirst)
    return integrator.k[1] = integrator.fsalfirst
end



@doc raw"""
    RKMK4

The Lie group variant of fourth-order Runge-Kutta algorithm for problems in the
[`LieODEProblemType`](@ref) formulation, called Runge-Kutta Munthe-Kaas.
The Butcher tableau is:

```math
\begin{array}{c|cccc}
0 & 0 \\
\frac{1}{2} & 0 & \frac{1}{2} & 0 \\
\frac{1}{2} & \frac{1}{2} & 0 \\
1 & 0 & 0 & 1 & 0\\
\hline
& \frac{1}{6} & \frac{1}{3} & \frac{1}{6} & \frac{1}{6}
\end{array}
```

For more details see [MuntheKaasOwren:1999](@cite).
"""
struct RKMK4{TR<:AbstractRetractionMethod,TG<:GroupAction} <:
       AbstractManifoldDiffEqAlgorithm
    retraction_method::TR
    action::TG
end

alg_order(::RKMK4) = 4

"""
    RKMK4Cache

Mutable cache for [`RKMK4`](@ref).
"""
struct RKMK4Cache{TID<:Identity} <: OrdinaryDiffEqMutableCache
    id::TID
end

"""
    RKMK4ConstantCache

Constant cache for [`RKMK4`](@ref).
"""
struct RKMK4ConstantCache <: OrdinaryDiffEqConstantCache end

function alg_cache(
    alg::RKMK4,
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
    return RKMK4Cache(Identity(base_lie_group(alg.action)))
end

function perform_step!(integrator, cache::RKMK4Cache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p, alg = integrator
    action = alg.action
    G = base_lie_group(action)
    M = base_manifold(action)
    k₁ = dt * f(u, p, t)
    Q₁ = k₁
    u₂ = Q₁ / 2
    k₂ =
        dt * f(
            retract(M, u, diff_group_apply(action, cache.id, u₂, u), alg.retraction_method),
            p,
            t + dt / 2,
        )
    Q₂ = k₂ - k₁
    u₃ = Q₁ / 2 + Q₂ / 2 - lie_bracket(G, Q₁, Q₂) / 8
    k₃ =
        dt * f(
            retract(M, u, diff_group_apply(action, cache.id, u₃, u), alg.retraction_method),
            p,
            t + dt / 2,
        )
    Q₃ = k₃ - k₂
    u₄ = Q₁ + Q₂ + Q₃
    k₄ =
        dt * f(
            retract(M, u, diff_group_apply(action, cache.id, u₄, u), alg.retraction_method),
            p,
            t + dt,
        )
    Q₄ = k₄ - 2 * k₂ + k₁
    v = Q₁ + Q₂ + Q₃ / 3 + Q₄ / 6 - lie_bracket(G, Q₁, Q₂) / 6 - lie_bracket(G, Q₁, Q₄) / 12

    X = diff_group_apply(action, cache.id, v, u)
    retract!(alg.manifold, u, u, X, alg.retraction_method)

    return integrator.stats.nf += 1
end

function initialize!(integrator, cache::RKMK4Cache)
    @unpack t, uprev, f, p, alg = integrator
    action = alg.action
    M = base_manifold(action)
    X = f(uprev, p, t) # Pre-start fsal
    integrator.fsalfirst =
        copy(M, diff_group_apply(action, cache.id, X, uprev))
    integrator.stats.nf += 1
    integrator.kshortsize = 1
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)
    # Avoid undefined entries if k is an array of arrays
    integrator.fsallast = zero.(integrator.fsalfirst)
    return integrator.k[1] = integrator.fsalfirst
end
