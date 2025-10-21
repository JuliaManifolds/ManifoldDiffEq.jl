module ManifoldDiffEq

import ConstructionBase: constructorof

using LinearAlgebra

using LieGroups
using LieGroups: AbstractLieGroup, GroupAction, base_lie_group, diff_group_apply

using Manifolds
using ManifoldsBase: retract_fused, retract_fused!, base_manifold


using Accessors: @set

using SciMLBase: isinplace, promote_tspan, solve!

using SciMLBase:
    AbstractODEFunction,
    AbstractODEProblem,
    CallbackSet,
    NullParameters,
    ODEFunction,
    ReturnCode,
    SciMLBase

using SciMLOperators: AbstractSciMLOperator

import SciMLBase: alg_order, isadaptive, solution_new_retcode, solve

using DiffEqBase:
    DiffEqBase,
    ODE_DEFAULT_NORM,
    ODE_DEFAULT_ISOUTOFDOMAIN,
    ODE_DEFAULT_UNSTABLE_CHECK,
    ODE_DEFAULT_PROG_MESSAGE

using OrdinaryDiffEqCore:
    default_controller,
    fsal_typeof,
    gamma_default,
    get_differential_vars,
    handle_dt!,
    initialize_callbacks!,
    initialize_d_discontinuities,
    initialize_saveat,
    initialize_tstops,
    isdtchangeable,
    qmax_default,
    qmin_default,
    qsteady_max_default,
    qsteady_min_default,
    uses_uprev

import OrdinaryDiffEqCore: perform_step!

using OrdinaryDiffEqCore:
    DEOptions,
    ODEIntegrator,
    OrdinaryDiffEqAlgorithm,
    OrdinaryDiffEqConstantCache,
    OrdinaryDiffEqCore,
    OrdinaryDiffEqMutableCache

using RecursiveArrayTools

using SimpleUnPack

"""
    abstract type AbstractManifoldDiffEqAlgorithm end

A subtype of `OrdinaryDiffEqAlgorithm` for manifold-aware algorithms.
"""
abstract type AbstractManifoldDiffEqAlgorithm <: OrdinaryDiffEqAlgorithm end

isadaptive(::AbstractManifoldDiffEqAlgorithm) = false

"""
    AbstractManifoldDiffEqAdaptiveAlgorithm <: AbstractManifoldDiffEqAlgorithm

An abstract subtype of `AbstractManifoldDiffEqAlgorithm` for adaptive algorithms.
This is the manifold-aware analogue of `OrdinaryDiffEqAdaptiveAlgorithm`.

"""
abstract type AbstractManifoldDiffEqAdaptiveAlgorithm <: AbstractManifoldDiffEqAlgorithm end

isadaptive(::AbstractManifoldDiffEqAdaptiveAlgorithm) = true

"""
    struct ManifoldODESolution{T} end

Counterpart of `SciMLBase.ODESolution`. It doesn't use the `N` parameter (because it
is not a generic manifold concept) and fields `u_analytic`, `errors`, `alg_choice`,
`original`, `tslocation` and `resid` (because we don't use them currently in
`ManifoldDiffEq.jl`).

Type parameter `T` denotes scalar floating point type of the solution

Fields:
* `u`: the representation of the ODE solution. Uses a nested power manifold representation.
* `t`: time points at which values in `u` were calculated.
* `k`: the representation of the `f` function evaluations at time points `k`. Uses a nested
  power manifold representation.
* `prob`: original problem that was solved.
* `alg`: [`AbstractManifoldDiffEqAlgorithm`](@ref) used to obtain the solution.
* `interp` [`ManifoldInterpolationData`](@ref). It is used for calculating solution values
  at times `t` other then the ones at which it was saved.
* `dense`: `true` if ODE solution is saved at every step and `false` otherwise.
* `stats`: [`DEStats`](https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/#SciMLBase.DEStats) of the solver
* `retcode`: [`ReturnCode`](https://docs.sciml.ai/SciMLBase/stable/interfaces/Solutions/#retcodes) of the solution.
"""
struct ManifoldODESolution{
        T <: Number,
        uType,
        tType,
        rateType,
        P,
        A <: AbstractManifoldDiffEqAlgorithm,
        IType,
        S,
    }
    u::uType
    t::tType
    k::rateType
    prob::P
    alg::A
    interp::IType
    dense::Bool
    stats::S
    retcode::ReturnCode.T
end

function ManifoldODESolution{T}(
        u,
        t,
        k,
        prob,
        alg,
        interp,
        dense,
        stats,
        retcode,
    ) where {T <: Number}
    return ManifoldODESolution{
        T,
        typeof(u),
        typeof(t),
        typeof(k),
        typeof(prob),
        typeof(alg),
        typeof(interp),
        typeof(stats),
    }(
        u,
        t,
        k,
        prob,
        alg,
        interp,
        dense,
        stats,
        retcode,
    )
end

constructorof(::Type{<:ManifoldODESolution{T}}) where {T <: Number} = ManifoldODESolution{T}

function solution_new_retcode(sol::ManifoldODESolution, retcode)
    return @set sol.retcode = retcode
end

function (sol::ManifoldODESolution)(
        t,
        ::Type{deriv} = Val{0};
        idxs = nothing,
        continuity = :left,
    ) where {deriv}
    return sol(t, deriv, idxs, continuity)
end

function (sol::ManifoldODESolution)(
        t::Number,
        ::Type{deriv},
        idxs::Nothing,
        continuity,
    ) where {deriv}
    return sol.interp(t, idxs, deriv, sol.prob.p, continuity)
end

include("utils.jl")
include("error_estimation.jl")
include("operators.jl")
include("interpolation.jl")
include("problems.jl")

alg_extrapolates(::AbstractManifoldDiffEqAlgorithm) = false

struct DefaultInit end


# Adapted from OrdinaryDiffEq.jl:
# https://github.com/SciML/OrdinaryDiffEq.jl/blob/1eef9db17600766bb71e7dce0cb105ae5f99b2a5/lib/OrdinaryDiffEqCore/src/solve.jl#L11
function SciMLBase.__init(
        prob::ManifoldODEProblem,
        alg::AbstractManifoldDiffEqAlgorithm,
        timeseries_init = (),
        ts_init = (),
        ks_init = (),
        recompile::Type{Val{recompile_flag}} = Val{true};
        saveat = (),
        tstops = (),
        d_discontinuities = (),
        save_everystep = isempty(saveat),
        save_on = true,
        save_start = save_everystep ||
            isempty(saveat) ||
            saveat isa Number ||
            prob.tspan[1] in saveat,
        save_end = nothing,
        callback = nothing,
        dense::Bool = save_everystep && isempty(saveat),
        calck = (callback !== nothing && callback !== CallbackSet()) ||
            (dense) ||
            !isempty(saveat), # and no dense output
        dt = eltype(prob.tspan)(0),
        dtmin = eltype(prob.tspan)(0),
        dtmax = eltype(prob.tspan)((prob.tspan[end] - prob.tspan[1])),
        force_dtmin = false,
        adaptive = isadaptive(alg),
        gamma = gamma_default(alg),
        abstol::Union{Nothing, Real} = nothing,
        reltol::Union{Nothing, Real} = nothing,
        qmin = qmin_default(alg),
        qmax = qmax_default(alg),
        qsteady_min = qsteady_min_default(alg),
        qsteady_max = qsteady_max_default(alg),
        qoldinit = isadaptive(alg) ? 1 // 10^4 : 0,
        controller = nothing,
        failfactor = 2,
        maxiters::Int = isadaptive(alg) ? 1000000 : typemax(Int),
        internalopnorm = opnorm,
        isoutofdomain = ODE_DEFAULT_ISOUTOFDOMAIN,
        unstable_check = ODE_DEFAULT_UNSTABLE_CHECK,
        verbose = true,
        timeseries_errors = true,
        dense_errors = false,
        advance_to_tstop = false,
        stop_at_next_tstop = false,
        initialize_save = true,
        progress = false,
        progress_steps = 1000,
        progress_name = "ODE",
        progress_message = ODE_DEFAULT_PROG_MESSAGE,
        progress_id = :OrdinaryDiffEq,
        userdata = nothing,
        allow_extrapolation = alg_extrapolates(alg),
        initialize_integrator = true,
        initializealg = DefaultInit(),
        kwargs...,
    ) where {recompile_flag}
    isdae = false

    if !isempty(saveat) && dense
        @warn(
            "Dense output is incompatible with saveat. Please use the SavingCallback from the Callback Library to mix the two behaviors."
        )
    end

    tType = eltype(prob.tspan)
    tspan = prob.tspan
    tdir = sign(tspan[end] - tspan[1])

    t = tspan[1]

    _alg = alg
    f = prob.f
    p = prob.p

    # Get the control variables

    M = prob.manifold

    u = copy(M, prob.u0)

    du = nothing
    duprev = nothing

    uType = typeof(u)
    uBottomEltype = recursive_bottom_eltype(u)
    uBottomEltypeNoUnits = recursive_unitless_bottom_eltype(u)

    uEltypeNoUnits = recursive_unitless_eltype(u)
    tTypeNoUnits = typeof(one(tType))

    if abstol === nothing
        abstol_internal = real(convert(uBottomEltype, oneunit(uBottomEltype) * 1 // 10^6))
    else
        abstol_internal = real(abstol)
    end

    if reltol === nothing
        reltol_internal = real(convert(uBottomEltype, oneunit(uBottomEltype) * 1 // 10^3))
    else
        reltol_internal = real(reltol)
    end

    dtmax > zero(dtmax) && tdir < 0 && (dtmax *= tdir) # Allow positive dtmax, but auto-convert
    # dtmin is all abs => does not care about sign already.

    if isinplace(prob) &&
            u isa AbstractArray &&
            eltype(u) <: Number &&
            uBottomEltypeNoUnits == uBottomEltype &&
            tType == tTypeNoUnits # Could this be more efficient for other arrays?
        rate_prototype = copy(M, u)
    else
        if (uBottomEltypeNoUnits == uBottomEltype && tType == tTypeNoUnits) ||
                eltype(u) <: Enum
            rate_prototype = u
        else # has units!
            rate_prototype = u / oneunit(tType)
        end
    end
    rateType = typeof(rate_prototype) ## Can be different if united

    tstops_internal = initialize_tstops(tType, tstops, d_discontinuities, tspan)
    saveat_internal = initialize_saveat(tType, saveat, tspan)
    d_discontinuities_internal =
        initialize_d_discontinuities(tType, d_discontinuities, tspan)

    callbacks_internal = CallbackSet(callback)

    max_len_cb = DiffEqBase.max_vector_callback_length_int(callbacks_internal)
    if max_len_cb !== nothing
        uBottomEltypeReal = real(uBottomEltype)
        if isinplace(prob)
            callback_cache = DiffEqBase.CallbackCache(
                u,
                max_len_cb,
                uBottomEltypeReal,
                uBottomEltypeReal,
            )
        else
            callback_cache =
                DiffEqBase.CallbackCache(max_len_cb, uBottomEltypeReal, uBottomEltypeReal)
        end
    else
        callback_cache = nothing
    end

    ### Algorithm-specific defaults ###
    ksEltype = Vector{rateType}

    # Have to convert in case passed in wrong.
    timeseries = timeseries_init === () ? uType[] : convert(Vector{uType}, timeseries_init)

    ts = ts_init === () ? tType[] : convert(Vector{tType}, ts_init)
    ks = ks_init === () ? ksEltype[] : convert(Vector{ksEltype}, ks_init)

    if (!adaptive || !isadaptive(_alg)) && save_everystep && tspan[2] - tspan[1] != Inf
        if dt == 0
            steps = length(tstops)
        else
            # For fixed dt, the only time dtmin makes sense is if it's smaller than eps().
            # Therefore user specified dtmin doesn't matter, but we need to ensure dt>=eps()
            # to prevent infinite loops.
            abs(dt) < dtmin && throw(ArgumentError("Supplied dt is smaller than dtmin"))
            steps = ceil(Int, abs((tspan[2] - tspan[1]) / dt))
        end
        sizehint!(timeseries, steps + 1)
        sizehint!(ts, steps + 1)
        sizehint!(ks, steps + 1)
    elseif save_everystep
        sizehint!(timeseries, 50)
        sizehint!(ts, 50)
        sizehint!(ks, 50)
    elseif !isempty(saveat_internal)
        savelength = length(saveat_internal) + 1
        if save_start == false
            savelength -= 1
        end
        if save_end == false && prob.tspan[2] in saveat_internal.valtree
            savelength -= 1
        end
        sizehint!(timeseries, savelength)
        sizehint!(ts, savelength)
        sizehint!(ks, savelength)
    else
        sizehint!(timeseries, 2)
        sizehint!(ts, 2)
        sizehint!(ks, 2)
    end

    QT = number_eltype(u)
    EEstT = number_eltype(u)

    k = rateType[]

    if uses_uprev(_alg, adaptive) || calck
        uprev = copy(M, u)
    else
        # Some algorithms do not use `uprev` explicitly. In that case, we can save
        # some memory by aliasing `uprev = u`, e.g. for "2N" low storage methods.
        uprev = u
    end
    if allow_extrapolation
        uprev2 = copy(M, u)
    else
        uprev2 = uprev
    end

    cache = alg_cache(
        _alg,
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
        reltol_internal,
        p,
        calck,
        Val(isinplace(prob)),
    )

    # Setting up the step size controller
    if controller === nothing
        controller = default_controller(_alg, cache, qoldinit, nothing, nothing)
    end

    save_end_user = save_end
    save_end =
        save_end === nothing ?
        save_everystep || isempty(saveat) || saveat isa Number || prob.tspan[2] in saveat :
        save_end

    save_idxs = nothing
    internalnorm = ODE_DEFAULT_NORM
    save_discretes = true

    opts = DEOptions{
        typeof(abstol_internal), typeof(reltol_internal),
        QT, tType, typeof(controller),
        typeof(internalnorm), typeof(internalopnorm),
        typeof(save_end_user),
        typeof(callbacks_internal),
        typeof(isoutofdomain),
        typeof(progress_message), typeof(unstable_check),
        typeof(tstops_internal),
        typeof(d_discontinuities_internal), typeof(userdata),
        typeof(save_idxs),
        typeof(maxiters), typeof(tstops),
        typeof(saveat), typeof(d_discontinuities),
    }(
        maxiters, save_everystep,
        adaptive, abstol_internal,
        reltol_internal,
        QT(gamma), QT(qmax),
        QT(qmin),
        QT(qsteady_max),
        QT(qsteady_min),
        QT(qoldinit),
        QT(failfactor),
        tType(dtmax), tType(dtmin),
        controller,
        internalnorm,
        internalopnorm,
        save_idxs, tstops_internal,
        saveat_internal,
        d_discontinuities_internal,
        tstops, saveat,
        d_discontinuities,
        userdata, progress,
        progress_steps,
        progress_name,
        progress_message,
        progress_id,
        timeseries_errors,
        dense_errors, dense,
        save_on, save_start,
        save_end, save_discretes, save_end_user,
        callbacks_internal,
        isoutofdomain,
        unstable_check,
        verbose, calck, force_dtmin,
        advance_to_tstop,
        stop_at_next_tstop
    )

    stats = SciMLBase.DEStats(0)
    differential_vars = get_differential_vars(f, u)


    manifold_interp = ManifoldInterpolationData(f, timeseries, ts, ks, dense, cache, M)
    sol = build_solution(
        prob,
        _alg,
        ts,
        timeseries,
        dense = dense,
        k = ks,
        manifold_interp = manifold_interp,
        stats = stats,
    )

    if recompile_flag == true
        FType = typeof(f)
        SolType = typeof(sol)
        cacheType = typeof(cache)
    else
        FType = Function
        SolType = DiffEqBase.AbstractDAESolution
        cacheType = DAECache
    end

    # rate/state = (state/time)/state = 1/t units, internalnorm drops units
    # we don't want to differentiate through eigenvalue estimation
    eigen_est = inv(one(tType))
    tprev = t
    dtcache = tType(dt)
    dtpropose = tType(dt)
    iter = 0
    kshortsize = 0
    reeval_fsal = false
    u_modified = false
    EEst = EEstT(1)
    just_hit_tstop = false
    isout = false
    accept_step = false
    force_stepfail = false
    last_stepfail = false
    do_error_check = true
    event_last_time = 0
    vector_event_last_time = 1
    last_event_error = zero(uBottomEltypeNoUnits)
    dtchangeable = isdtchangeable(_alg)
    q11 = QT(1)
    success_iter = 0
    erracc = QT(1)
    dtacc = tType(1)
    reinitiailize = true
    saveiter = 0 # Starts at 0 so first save is at 1
    saveiter_dense = 0
    #fsalfirst, fsallast = get_fsalfirstlast(cache, rate_prototype)
    fsalfirst, fsallast = allocate(rate_prototype), allocate(rate_prototype)

    integrator = ODEIntegrator{
        typeof(_alg),
        isinplace(prob),
        uType,
        typeof(du),
        tType,
        typeof(p),
        typeof(eigen_est),
        typeof(EEst),
        QT,
        typeof(tdir),
        typeof(k),
        SolType,
        FType,
        cacheType,
        typeof(opts),
        typeof(fsalfirst),
        typeof(last_event_error),
        typeof(callback_cache),
        typeof(initializealg),
        typeof(differential_vars),
    }(
        sol,
        u,
        du,
        k,
        t,
        tType(dt),
        f,
        p,
        uprev,
        uprev2,
        duprev,
        tprev,
        _alg,
        dtcache,
        dtchangeable,
        dtpropose,
        tdir,
        eigen_est,
        EEst,
        QT(qoldinit),
        q11,
        erracc,
        dtacc,
        success_iter,
        iter,
        saveiter,
        saveiter_dense,
        cache,
        callback_cache,
        kshortsize,
        force_stepfail,
        last_stepfail,
        just_hit_tstop,
        do_error_check,
        event_last_time,
        vector_event_last_time,
        last_event_error,
        accept_step,
        isout,
        reeval_fsal,
        u_modified,
        reinitiailize,
        isdae,
        opts,
        stats,
        initializealg,
        differential_vars,
        fsalfirst,
        fsallast,
    )

    if initialize_integrator
        if SciMLBase.has_initializeprob(prob.f)
            update_uprev!(integrator)
        end

        if save_start
            integrator.saveiter += 1 # Starts at 1 so first save is at 2
            integrator.saveiter_dense += 1
            copyat_or_push!(ts, 1, t)
            # N.B.: integrator.u can be modified by initialized_dae!
            copyat_or_push!(timeseries, 1, integrator.u)
            copyat_or_push!(ks, 1, [rate_prototype])
        else
            integrator.saveiter = 0 # Starts at 0 so first save is at 1
            integrator.saveiter_dense = 0
        end

        initialize_callbacks!(integrator, initialize_save)
        initialize!(integrator, integrator.cache)

    end

    handle_dt!(integrator)
    return integrator
end

function solve(
        prob::ManifoldODEProblem,
        alg::AbstractManifoldDiffEqAlgorithm,
        args...;
        u0 = nothing,
        p = nothing,
        kwargs...,
    )
    u0 = u0 !== nothing ? u0 : prob.u0
    p = p !== nothing ? p : prob.p

    integrator = SciMLBase.__init(prob, alg, args...; kwargs...)
    solve!(integrator)
    return integrator.sol
end

include("manifold_solvers.jl")
include("frozen_solvers.jl")
include("lie_solvers.jl")

export alg_order, solve

export FrozenManifoldDiffEqOperator, LieManifoldDiffEqOperator, ManifoldODEProblem

end # module
