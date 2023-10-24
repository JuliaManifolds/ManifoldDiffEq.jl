
@doc raw"""
    LieODEProblemType

An initial value problem manifold ordinary differential equation in the Lie action formulation.

A Lie ODE on manifold ``M`` is defined in terms a vector field ``F: (ℝ × P × M) \to 𝔤``
where ``𝔤`` is the Lie algebra of a Lie group ``G`` acting on ``M``, with an
initial value ``y₀`` and ``P`` is the space of constant parameters. A solution to this
problem is a curve ``y:ℝ\to M`` such that ``y(0)=y₀`` and for each ``t ∈ [0, T]`` we have
``D_t y(t) = F(y(t), p, t)∘y(t)``, where the ``∘`` is defined as
````math
X∘m = \frac{d}{dt}\vert_{t=0} \exp(tZ)⋅m
````
and ``⋅`` is the group action of ``G`` on ``M``.

!!! note

    Proofs of convergence and order have several assumptions, including time-independence
    of ``F``. Integrators may not work well if these assumptions do not hold.
"""
struct LieODEProblemType end

@doc raw"""
    ExplicitManifoldODEProblemType

An initial value problem manifold ordinary differential equation in the frozen coefficients
formulation by Crouch and Grossman, see [CrouchGrossman:1993](@cite).

A frozen coefficients ODE on manifold ``M`` is defined in terms a vector field
``F: (M × P × ℝ) \to T_p M`` where ``p`` is the point given as the third argument to ``F``,
with an initial value ``y₀`` and ``P`` is the space of constant parameters. A solution to
this problem is a curve ``y:ℝ\to M`` such that ``y(0)=y₀`` and for each ``t ∈ [0, T]`` we
have ``D_t y(t) = F(y(t), p, t)``,

!!! note

    Proofs of convergence and order have several assumptions, including time-independence
    of ``F``. Integrators may not work well if these assumptions do not hold.

"""
struct ExplicitManifoldODEProblemType end


"""
    ManifoldODEProblem

A general problem for ODE problems on Riemannian manifolds.

# Fields

* `f` the tangent vector field `f(u,p,t)`
* `u0` the initial condition
* `tspan` time interval for the solution
* `p` constant parameters for `f``
* `kwargs` A callback to be applied to every solver which uses the problem.
* `problem_type` type of problem
* `manifold` the manifold the vector field is defined on
"""
struct ManifoldODEProblem{uType,tType,isinplace,P,F,K,PT,TM} <:
       AbstractODEProblem{uType,tType,isinplace}
    f::F
    u0::uType
    tspan::tType
    p::P
    kwargs::K
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
    return ManifoldODEProblem(ODEFunction(f), u0, tspan, manifold, p; kwargs...)
end

function OrdinaryDiffEq.ode_determine_initdt(
    u0,
    t,
    tdir,
    dtmax,
    abstol,
    reltol,
    internalnorm,
    prob::ManifoldODEProblem{uType,tType},
    integrator,
) where {tType,uType}
    _tType = number_eltype(tType)
    oneunit_tType = oneunit(_tType)
    return convert(_tType, oneunit_tType * 1 // 10^(6))
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
    retcode = ReturnCode.Default,
    stats = nothing,
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
    return ODESolution{T,1}(
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
        stats,
        nothing,
        retcode,
    )
end
