# Frozen coefficients solvers

An initial value problem manifold ordinary differential equation in the frozen coefficients
formulation by Crouch and Grossman, see [CrouchGrossman:1993](@cite).

A frozen coefficients ODE on manifold ``M`` is defined in terms a vector field
``F\colon (M × P × ℝ) → T_p M`` where ``p`` is the point given as the third argument to ``F``,
with an initial value ``y_0`` and ``P`` is the space of constant parameters.
Frozen coefficients mean that we also have means to transport a vector ``X \in T_p M`` obtained
from ``F`` to a different point on a manifold or a different time (parameters are assumed
to be constant). This is performed through `operator_vector_transport`, an object of
a subtype of [`AbstractVectorTransportOperator`](@ref ManifoldDiffEq.AbstractVectorTransportOperator), stored in [`FrozenManifoldDiffEqOperator`](@ref ManifoldDiffEq.FrozenManifoldDiffEqOperator).

A solution to this problem is a curve ``y\colon ℝ\to M`` such that ``y(0)=y_0`` and for each
``t \in [0, T]`` we have ``D_t y(t) = F(y(t), p, t)``.

The problem is usually studied for manifolds that are Lie groups or homogeneous manifolds, see[CelledoniMarthinsenOwren:2014](@cite).

Note that in this formulation ``s``-stage explicit Runge-Kutta schemes that for ``ℝ^n`` are defined by equations

````math
\begin{align*}
X_1 &= f(u_n, p, t) \\
X_2 &= f(u_n+h a_{2,1} X_1, p, t+c_2 h) \\
X_3 &= f(u_n+h a_{3,1} X_1 + a_{3,2} X_2, p, t+c_3 h) \\
&\vdots \\
X_s &= f(u_n+h a_{s,1} X_1 + a_{s,2} X_2 + \dots + a_{s,s-1} X_{s-1}, p, t+c_s h) \\
u_{n+1} &= u_n + h\sum_{i=1}^s b_i X_i
\end{align*}
````

for general manifolds read

````math
\begin{align*}
X_1 &= f(u_n, p, t) \\
u_{n,2,1} &= \exp_{u_n}(h a_{2,1} X_1) \\
X_2 &= f(u_{n,2,1}, p, t+c_2 h) \\
u_{n,3,1} &= \exp_{u_n}(h a_{3,1} X_1) \\
u_{n,3,2} &= \exp_{u_{n,3,1}}(\mathcal P_{u_{n,3,1}\gets u_{n,2,1}} h a_{3,2} X_2) \\
X_3 &= f(u_{n,3,2}, p, t+c_3 h) \\
&\vdots \\
X_s &= f(u_{n,s,s-1}, p, t+c_s h) \\
X_{b,1} &= X_1 \\
u_{b,1} &= \exp_{u_n}(h b_1 X_{b,1}) \\
X_{b,2} &= \mathcal P_{u_{b,1} \gets u_{n,2,1}} X_2 \\
u_{b,2} &= \exp_{u_{b,1}}(h b_2 X_{b,2}) \\
&\vdots \\
X_{b,s} &= \mathcal P_{u_{b,s-1} \gets u_{n,s,s-1}} X_s \\
u_{n+1} &= \exp_{u_{b,s-1}}(h b_s X_{b,s})
\end{align*}
````

Vector transports correspond to handling frozen coefficients. Note that the implementation allows for easy substitution of methods used for calculation of the exponential map (for example to use an approximation) and vector transport (if the default vector transport is not suitable for the problem). It is desirable to use a flat vector transport instead of a torsion-free one when available, for example the plus or minus Cartan-Schouten connections on Lie groups.

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

## Literature
