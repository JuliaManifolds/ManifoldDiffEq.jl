# Frozen coefficients solvers

An initial value problem manifold ordinary differential equation in the frozen coefficients
formulation by Crouch and Grossman, see [^CrouchGrossman1993].

A frozen coefficients ODE on manifold ``M`` is defined in terms a vector field
``F\colon (M × P × ℝ) \to T_p M`` where ``p`` is the point given as the third argument to ``F``,
with an initial value ``y_0`` and ``P`` is the space of constant parameters.
Frozen coefficients mean that we also have means to transport a vector ``X \in T_p M`` obtained
from ``F`` to a different point on a manifold or a different time (parameters are assumed
to be constant). This is performed through `operator_vector_transport`, an object of
a subtype of [`AbstractVectorTransportOperator`](@ref ManifoldDiffEq.AbstractVectorTransportOperator), stored in [`FrozenManifoldDiffEqOperator`](@ref ManifoldDiffEq.FrozenManifoldDiffEqOperator).

A solution to this problem is a curve ``y\colon ℝ\to M`` such that ``y(0)=y_0`` and for each
``t \in [0, T]`` we have ``D_t y(t) = F(y(t), p, t)``.

The problem is usually studied for manifolds that are Lie groups or homogeneous manifolds, see[^CelledoniMarthinsenOwren2014].

[^CrouchGrossman1993]:
    > P. E. Crouch and R. Grossman, “Numerical integration of ordinary differential
    > equations on manifolds,” J Nonlinear Sci, vol. 3, no. 1, pp. 1–33, Dec. 1993,
    > doi: 10.1007/BF02429858.

[^CelledoniMarthinsenOwren2014]:
    > E. Celledoni, H. Marthinsen, and B. Owren, “An introduction to Lie group integrators -- basics, new developments and applications,” Journal of Computational Physics, vol. 257, pp. 1040–1061, Jan. 2014, doi: 10.1016/j.jcp.2012.12.031.


```@autodocs
Modules = [ManifoldDiffEq]
Pages = ["frozen_solvers.jl"]
Order = [:type, :function]
```

```@docs
ManifoldDiffEq.ExplicitManifoldODEProblemType
ManifoldDiffEq.FrozenManifoldDiffEqOperator
ManifoldDiffEq.AbstractVectorTransportOperator
ManifoldDiffEq.DefaultVectorTransportOperator
```
