


@doc raw"""
    apply_diff_group(A::AbstractGroupAction, a, X, p)

For a point on manifold ``p ∈ \mathcal M`` and an element `X` of the tangent space at `a`,
an element of the Lie group of action `A`, ``X ∈ T_a \mathcal G``, compute the
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
    retraction_method::TR
    action::TG
end

alg_order(::ManifoldLieEuler) = 1

"""
    ManifoldLieEulerCache

Cache for [`ManifoldLieEuler`](@ref).
"""
struct ManifoldLieEulerCache{TID<:Identity} <: OrdinaryDiffEqMutableCache
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
    return ManifoldLieEulerCache(Identity(base_group(alg.action)))
end

function perform_step!(integrator, cache::ManifoldLieEulerCache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p, alg = integrator

    X = f(u, p, t)
    action = alg.action
    k = apply_diff_group(action, cache.id, X, u)

    retract!(alg.manifold, u, u, dt * k, alg.retraction_method)

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



@doc raw"""
    RKMK4

The Lie group variant of fourth-order Runge-Kutta algorithm for problems in the
[`LieODEProblemType`](@ref) formulation, called Runge-Kutta Munthe-Kaas.
The Nutcher tableau is:

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

For more details see [^MuntheKaasOwren1999].

[^MuntheKaasOwren1999]:
    > H. Munthe–Kaas and B. Owren, “Computations in a free Lie algebra,” Philosophical
    > Transactions of the Royal Society of London. Series A: Mathematical, Physical and
    > Engineering Sciences, vol. 357, no. 1754, pp. 957–981, Apr. 1999,
    > doi: [10.1098/rsta.1999.0361](https://doi.org/10.1098/rsta.1999.0361).
"""
struct RKMK4{TM<:AbstractManifold,TR<:AbstractRetractionMethod,TG<:AbstractGroupAction} <:
       OrdinaryDiffEqAlgorithm
    manifold::TM
    retraction_method::TR
    action::TG
end

alg_order(::RKMK4) = 4

"""
    RKMK4Cache

Cache for [`RKMK4`](@ref).
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
    return RKMK4Cache(Identity(base_group(alg.action)))
end

function perform_step!(integrator, cache::RKMK4Cache, repeat_step = false)
    @unpack t, dt, uprev, u, f, p, alg = integrator

    action = alg.action
    G = base_group(action)
    M = alg.manifold
    k₁ = dt * f(u, p, t)
    Q₁ = k₁
    u₂ = Q₁ / 2
    k₂ =
        dt * f(
            retract(M, u, apply_diff_group(action, cache.id, u₂, u), alg.retraction_method),
            p,
            t + dt / 2,
        )
    Q₂ = k₂ - k₁
    u₃ = Q₁ / 2 + Q₂ / 2 - lie_bracket(G, Q₁, Q₂) / 8
    k₃ =
        dt * f(
            retract(M, u, apply_diff_group(action, cache.id, u₃, u), alg.retraction_method),
            p,
            t + dt / 2,
        )
    Q₃ = k₃ - k₂
    u₄ = Q₁ + Q₂ + Q₃
    k₄ =
        dt * f(
            retract(M, u, apply_diff_group(action, cache.id, u₄, u), alg.retraction_method),
            p,
            t + dt,
        )
    Q₄ = k₄ - 2 * k₂ + k₁
    v = Q₁ + Q₂ + Q₃ / 3 + Q₄ / 6 - lie_bracket(G, Q₁, Q₂) / 6 - lie_bracket(G, Q₁, Q₄) / 12

    X = apply_diff_group(action, cache.id, v, u)
    retract!(alg.manifold, u, u, X, alg.retraction_method)

    return integrator.destats.nf += 1
end

function initialize!(integrator, cache::RKMK4Cache)
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
