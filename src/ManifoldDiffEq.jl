module ManifoldDiffEq

using Manifolds
using SciMLBase
using OrdinaryDiffEq

using OrdinaryDiffEq:
    OrdinaryDiffEqAlgorithm,
    OrdinaryDiffEqMutableCache,
    OrdinaryDiffEqConstantCache,
    alg_order,
    trivial_limiter!,
    constvalue,
    @muladd,
    @unpack,
    @cache,
    @..

import OrdinaryDiffEq: alg_cache, initialize!, perform_step!


@doc raw"""
    LieODEProblemType

An initial value problem manifold ordinary differential equation in the Lie action formulation.

A Lie ODE on manifold ``M`` is defined in terms a vector field ``F: (ℝ × P × M) \to 𝔤``
where ``𝔤`` is the Lie algebra of a Lie group ``G`` acting on ``M``, with an
initial value ``y₀`` and ``P`` is the space of constant parameters. A solution to this
problem is a curve ``y:ℝ\to M`` such that ``y(0)=y₀`` and for each ``t ∈ [0, T]`` we have
``D_t y(t) = f(t, y(t))∘y(t)``, where the ``∘`` is defined as
````
    X⋅m = \frac{d}{dt}\vert_{t=0} \exp(tZ)⋅m
````
and ``⋅`` is the group action of ``G`` on ``M``.

!!! note

    Proofs of convergence and order have several assumptions, including time-independence
    of ``F``. Integrators may not work well if these assumptions do not hold.
"""
struct LieODEProblemType{TG<:AbstractGroupAction}
    action::TG
end

@doc raw"""
    ExplicitManifoldODEProblemType

An initial value problem manifold ordinary differential equation in the general formulation.
Can be used to express problems in the vector fields with frozen coefficients formulation by
Crouch and Grossman, see [^Crouch1993].

A Lie ODE on manifold ``M`` is defined in terms a vector field ``F: (ℝ × P × M) \to T_p M``
where ``p`` is the point given as the third argument to ``F``, with an
initial value ``y₀`` and ``P`` is the space of constant parameters. A solution to this
problem is a curve ``y:ℝ\to M`` such that ``y(0)=y₀`` and for each ``t ∈ [0, T]`` we have
``D_t y(t) = F(t, p, y(t))``,

!!! note

    Proofs of convergence and order have several assumptions, including time-independence
    of ``F``. Integrators may not work well if these assumptions do not hold.

[^Crouch1993]:
    > P. E. Crouch and R. Grossman, “Numerical integration of ordinary differential
    > equations on manifolds,” J Nonlinear Sci, vol. 3, no. 1, pp. 1–33, Dec. 1993,
    > doi: 10.1007/BF02429858.

"""
struct ExplicitManifoldODEProblemType end

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

    copyto!(u, retract(alg.manifold, u, dt * L(u, p, t), alg.retraction))

    integrator.f(integrator.fsallast, u, p, t + dt)
    return integrator.destats.nf += 1
end

function initialize!(integrator, cache::LieEulerCache)
    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
    integrator.destats.nf += 1
    integrator.kshortsize = 1
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    # Avoid undefined entries if k is an array of arrays
    integrator.fsallast = zero(integrator.fsalfirst)
    return integrator.k[1] = integrator.fsalfirst
end



#########

"""
    ManifoldDiffEqOperator{T<:Number,TF} <: AbstractDiffEqOperator{T}

DiffEq operator on manifolds.
"""
struct ManifoldDiffEqOperator{T<:Number,TF} <: SciMLBase.AbstractDiffEqOperator{T}
    func::TF
end

ManifoldDiffEqOperator{T}(f) where {T<:Number} = ManifoldDiffEqOperator{T,typeof(f)}(f)

function (L::ManifoldDiffEqOperator)(du, u, p, t)
    return copyto!(du, L.func(u, p, t))
end
function (L::ManifoldDiffEqOperator)(u, p, t)
    return L.func(u, p, t)
end


end # module
