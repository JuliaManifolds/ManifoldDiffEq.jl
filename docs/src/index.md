# ManifoldDiffEq

The package __ManifoldDiffEq__ aims to provide a library of differential equation solvers
on manifolds. The library is built on top of [`Manifolds.jl`](https://github.com/JuliaManifolds/Manifolds.jl) and [`LieGroups.jl`](https://github.com/JuliaManifolds/LieGroups.jl) and follows the interface of [`OrdinaryDiffEq.jl`](https://github.com/SciML/OrdinaryDiffEq.jl/).

The code in [Examples](@ref) demonstrates usage of __ManifoldDiffEq__ to solve a simple equation and visualize the results.

Methods implemented in this library are described for example in [HairerLubichWanner:2010](@cite).

```@docs
ManifoldDiffEq.ManifoldODEProblem
```

# Literature
