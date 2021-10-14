module ManifoldDiffEq

using Manifolds
using SciMLBase:
    SciMLBase, AbstractODEProblem, AbstractODEFunction, NullParameters, promote_tspan
import SciMLBase: build_solution
using OrdinaryDiffEq

using OrdinaryDiffEq:
    InterpolationData,
    OrdinaryDiffEqAlgorithm,
    OrdinaryDiffEqAdaptiveAlgorithm,
    OrdinaryDiffEqMutableCache,
    OrdinaryDiffEqConstantCache,
    trivial_limiter!,
    constvalue,
    @muladd,
    @unpack,
    @cache,
    @..

import OrdinaryDiffEq: alg_cache, alg_order, initialize!, perform_step!

using RecursiveArrayTools

include("utils.jl")
include("error_estimation.jl")
include("operators.jl")
include("problems.jl")
include("interpolation.jl")
include("frozen_solvers.jl")
include("lie_solvers.jl")

end # module
