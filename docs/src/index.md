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

using ManifoldDiffEq, OrdinaryDiffEq

A = ManifoldDiffEq.ManifoldDiffEqOperator{Float64}() do u, p, t
    return f2(u...)
end
prob = ODEProblem(A, [0.0, 1.0, 0.0], (0, 20.0))
alg = ManifoldDiffEq.ManifoldLieEuler(Sphere(2), ExponentialRetraction())
sol1 = solve(prob, alg, dt = 0.001)

Makie.lines!([u[1] for u in sol1.u], [u[2] for u in sol1.u], [u[3] for u in sol1.u]; linewidth = 10, color=:red)
```
