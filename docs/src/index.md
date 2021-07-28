# ManifoldDiffEq

The package __ManifoldDiffEq__ aims to provide a library of differential equation solvers
on manifolds. The library is built on top of [`Manifolds.jl`](https://github.com/JuliaManifolds/Manifolds.jl) and follows the interface of [`OrdinaryDiffEq.jl`](https://github.com/SciML/OrdinaryDiffEq.jl/).

The code below demonstrates osage of __ManifoldDiffEq__ to solve a simple equation and visualize the results.

```julia
using GLMakie, Makie, LinearAlgebra

n = 10

θ = [0;(0.5:n-0.5)/n;1]
φ = [(0:2n-2)*2/(2n-1);2]
x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
z = [cospi(θ) for θ in θ, φ in φ]

function f2(x, y, z)
    p = [x, y, z]
    return exp(x) * cross(p, [1.0, 0.0, 0.0]) + exp(y) * cross(p, [0.0, 1.0, 0.0])
end

tans = f2.(vec(x), vec(y), vec(z))
u = [a[1] for a in tans]
v = [a[2] for a in tans]
w = [a[3] for a in tans]

scene = Scene();

arr = Makie.arrows(
    vec(x), vec(y), vec(z), u, v, w;
    arrowsize = 0.1, linecolor = (:gray, 0.7), linewidth = 0.02, lengthscale = 0.1
)

using ManifoldDiffEq, OrdinaryDiffEq, Manifolds

# This is the same ODE problem on two different formulations: Lie group action (prob_lie)
# and frozen coefficients (prob_frozen)
S2 = Manifolds.Sphere(2)

A_lie = ManifoldDiffEq.ManifoldDiffEqOperator{Float64}() do u, p, t
    return hat(SpecialOrthogonal(3), Matrix(I(3)), cross(u, f2(u...)))
end
prob_lie = ODEProblem(A_lie, [0.0, 1.0, 0.0], (0, 20.0))

A_frozen = ManifoldDiffEq.FrozenManifoldDiffEqOperator{Float64}(S2) do u, p, t
    return f2(u...)
end
prob_frozen = ODEProblem(A_frozen, [0.0, 1.0, 0.0], (0, 20.0))

action = RotationAction(Euclidean(3), SpecialOrthogonal(3))
alg_lie_euler = ManifoldDiffEq.ManifoldLieEuler(S2, ExponentialRetraction(), action)
alg_lie_rkmk4 = ManifoldDiffEq.RKMK4(S2, ExponentialRetraction(), action)

alg_manifold_euler = ManifoldDiffEq.ManifoldEuler(S2, ExponentialRetraction())
alg_cg2 = ManifoldDiffEq.CG2(S2, ExponentialRetraction())
alg_cg3 = ManifoldDiffEq.CG3(S2, ExponentialRetraction())

dt = 0.2
sol_lie = solve(prob_lie, alg_lie_euler, dt = dt)
sol_frozen_cg2 = solve(prob_frozen, alg_cg2, dt = dt)
sol_frozen_cg3 = solve(prob_frozen, alg_cg3, dt = dt)
sol_rkmk4 = solve(prob_lie, alg_lie_rkmk4, dt = dt)

for (sol, color) in [(sol_lie, :red), (sol_frozen_cg2, :green), (sol_rkmk4, :blue)]
    Makie.lines!([u[1] for u in sol.u], [u[2] for u in sol.u], [u[3] for u in sol.u]; linewidth = 10, color=color)
end
```
