
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
    ManifoldODEProblem


"""
struct ManifoldODEProblem{uType,tType,isinplace,P,F,K,PT,TM} <:
       AbstractODEProblem{uType,tType,isinplace}
    f::F # The ODE is `du/dt = f(u,p,t)`.
    u0::uType # The initial condition is `u(tspan[1]) = u0`.
    tspan::tType # The solution `u(t)` will be computed for `tspan[1] ≤ t ≤ tspan[2]`.
    p::P # Constant parameters to be supplied as the second argument of `f`.
    kwargs::K # A callback to be applied to every solver which uses the problem.
    problem_type::PT
    manifold::TM
    function ManifoldODEProblem{iip}(
        f::AbstractODEFunction{iip},
        u0,
        tspan,
        manifold::AbstractManifold,
        p = NullParameters(),
        problem_type = ExplicitManifoldODEProblemType();
        kwargs...,
    ) where {iip}
        _tspan = promote_tspan(tspan)
        return new{
            typeof(u0),
            typeof(_tspan),
            isinplace(f),
            typeof(p),
            typeof(f),
            typeof(kwargs),
            typeof(problem_type),
            typeof(manifold),
        }(
            f,
            u0,
            _tspan,
            p,
            kwargs,
            problem_type,
            manifold,
        )
    end

    """
        ManifoldODEProblem{isinplace}(f,u0,tspan,p=NullParameters(),callback=CallbackSet())

    Define an ODE problem with the specified function.
    `isinplace` optionally sets whether the function is inplace or not.
    This is determined automatically, but not inferred.
    """
    function ManifoldODEProblem{iip}(
        f,
        u0,
        tspan,
        manifold::AbstractManifold,
        p = NullParameters();
        kwargs...,
    ) where {iip}
        return ManifoldODEProblem(
            convert(ODEFunction{iip}, f),
            u0,
            tspan,
            manifold,
            p;
            kwargs...,
        )
    end

    function ManifoldODEProblem{iip,recompile}(
        f,
        u0,
        tspan,
        manifold::AbstractManifold,
        p = NullParameters();
        kwargs...,
    ) where {iip,recompile}
        if !recompile
            if iip
                ManifoldODEProblem{iip}(
                    wrapfun_iip(f, (u0, u0, p, tspan[1])),
                    u0,
                    tspan,
                    manifold,
                    p;
                    kwargs...,
                )
            else
                ManifoldODEProblem{iip}(
                    wrapfun_oop(f, (u0, p, tspan[1])),
                    u0,
                    tspan,
                    manifold,
                    p;
                    kwargs...,
                )
            end
        else
            ManifoldODEProblem{iip}(f, u0, tspan, p, manifold; kwargs...)
        end
    end
end

"""
    ManifoldODEProblem(f::ODEFunction, u0, tspan, p=NullParameters(), callback=CallbackSet())

Define an ODE problem from an [`ODEFunction`](@ref).
"""
function ManifoldODEProblem(
    f::AbstractODEFunction,
    u0,
    tspan,
    manifold::AbstractManifold,
    args...;
    kwargs...,
)
    return ManifoldODEProblem{isinplace(f)}(f, u0, tspan, manifold, args...; kwargs...)
end

function ManifoldODEProblem(
    f,
    u0,
    tspan,
    manifold::AbstractManifold,
    p = NullParameters();
    kwargs...,
)
    return ManifoldODEProblem(convert(ODEFunction, f), u0, tspan, manifold, p; kwargs...)
end
