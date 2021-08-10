# ManifoldDiffEq

The package __ManifoldDiffEq__ aims to provide a library of differential equation solvers
on manifolds. The library is built on top of [`Manifolds.jl`](https://github.com/JuliaManifolds/Manifolds.jl) and follows the interface of [`OrdinaryDiffEq.jl`](https://github.com/SciML/OrdinaryDiffEq.jl/).

The code below demonstrates usage of __ManifoldDiffEq__ to solve a simple equation and visualize the results.

Methods implemented in this library are described for example in[^HairerLubichWanner2010].


[^HairerLubichWanner2010]:
    > E. Hairer, C. Lubich, and G. Wanner, Geometric Numerical Integration: Structure-Preserving
    > Algorithms for Ordinary Differential Equations, 2nd ed. 2006. 2nd printing 2010 edition. Heidelberg ;
    > New York: Springer, 2010.
    > doi: [10.1007/3-540-30666-8](https://doi.org/10.1007/3-540-30666-8).

```@docs
ManifoldDiffEq.ManifoldODEProblem
```

# Literature