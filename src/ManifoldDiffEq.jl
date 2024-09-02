module ManifoldDiffEq

using ManifoldsBase
using Manifolds

using SciMLBase: isinplace, promote_tspan

using SciMLBase:
    ReturnCode,
    NullParameters,
    AbstractDiffEqOperator,
    AbstractODEFunction,
    AbstractODEProblem,
    ODEFunction

import SciMLBase: alg_order, solve

using RecursiveArrayTools

using SimpleUnPack

"""
    abstract type AbstractManifoldDiffEqAlgorithm end

Counterpart of `OrdinaryDiffEqAlgorithm`.
"""
abstract type AbstractManifoldDiffEqAlgorithm end

"""
    abstract type AbstractManifoldDiffEqAdaptiveAlgorithm end

Counterpart of `OrdinaryDiffEqAdaptiveAlgorithm`.
"""
abstract type AbstractManifoldDiffEqAdaptiveAlgorithm <: AbstractManifoldDiffEqAlgorithm end

"""
    struct ManifoldODESolution end

Counterpart of `SciMLBase.ODESolution`. It doesn't use the `N` parameter.
"""
struct ManifoldODESolution{
    T,
    uType,
    uType2,
    DType,
    tType,
    rateType,
    P,
    A,
    IType,
    S,
    AC<:Union{Nothing,Vector{Int}},
    R,
    O,
}
    u::uType
    u_analytic::uType2
    errors::DType
    t::tType
    k::rateType
    prob::P
    alg::A
    interp::IType
    dense::Bool
    tslocation::Int
    stats::S
    alg_choice::AC
    retcode::ReturnCode.T
    resid::R
    original::O
end

include("utils.jl")
include("error_estimation.jl")
include("operators.jl")
include("problems.jl")


function solve(prob::ManifoldODEProblem, alg::AbstractManifoldDiffEqAlgorithm) end

include("interpolation.jl")
include("frozen_solvers.jl")
include("lie_solvers.jl")

export alg_order, solve

end # module
