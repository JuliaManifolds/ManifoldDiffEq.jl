using Test
using ManifoldDiffEq
using Manifolds
using OrdinaryDiffEq
using LinearAlgebra

@testset "ManifoldDiffEq" begin

    A = ManifoldDiffEq.ManifoldDiffEqOperator{Float64}() do u, p, t
        return cross(u, [1.0, 0.0, 0.0])
    end
    prob = ODEProblem(A, [0.0, 1.0, 0.0], (0, 2.0))
    alg = ManifoldDiffEq.ManifoldLieEuler(Sphere(2), ExponentialRetraction())
    sol1 = solve(
        prob,
        alg,
        dt = 1 / 8,
    )
    @test alg_order(alg) == 1

    @test sol1(0.0) ≈ [0.0, 1.0, 0.0]
end
