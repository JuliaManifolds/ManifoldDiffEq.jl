# Lie group solvers

An initial value problem manifold ordinary differential equation in the Lie action formulation.

A Lie ODE on manifold ``M`` is defined in terms a vector field ``F: (M × P × ℝ) \to 𝔤``
where ``𝔤`` is the Lie algebra of a Lie group ``G`` acting on ``M``, with an
initial value ``y_0`` and ``P`` is the space of constant parameters. A solution to this
problem is a curve ``y\colon ℝ\to M`` such that ``y(0)=y_0`` and for each ``t \in [0, T]`` we have
``D_t y(t) = f(y(t), p, t)\circ y(t)``, where the ``\circ`` is defined as
````math
X\circ m = \frac{d}{dt}\vert_{t=0} \exp(tZ)\cdot m
````
and ``\cdot`` is the group action of ``G`` on ``M``.

The Lie group ``G`` must act transitively on ``M``, that is for each pair of points ``p, q`` on ``M`` there is an element ``a \in G`` such that ``a\cdot p = q``. See for example [^CelledoniMarthinsenOwren2014] for details.

[^CelledoniMarthinsenOwren2014]:
    > E. Celledoni, H. Marthinsen, and B. Owren,
    > “An introduction to Lie group integrators -- basics, new developments and applications,”
    > Journal of Computational Physics, vol. 257, pp. 1040–1061, Jan. 2014,
    > doi: [10.1016/j.jcp.2012.12.031](https://doi.org/10.1016/j.jcp.2012.12.031).

```@autodocs
Modules = [ManifoldDiffEq]
Pages = ["lie_solvers.jl"]
Order = [:type, :function]
```

```@docs
ManifoldDiffEq.LieODEProblemType
ManifoldDiffEq.LieManifoldDiffEqOperator
```

## Literature
