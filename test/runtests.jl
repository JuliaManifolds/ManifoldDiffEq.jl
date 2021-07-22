using Test
using ManifoldDiffEq
using Manifolds
using OrdinaryDiffEq: OrdinaryDiffEq, alg_order
using LinearAlgebra

@testset "ManifoldDiffEq" begin

    @testset "Sphere" begin
        A = ManifoldDiffEq.ManifoldDiffEqOperator{Float64}() do u, p, t
            return cross(u, [1.0, 0.0, 0.0])
        end
        u0 = [0.0, 1.0, 0.0]
        M = Sphere(2)
        prob = ManifoldDiffEq.ManifoldODEProblem(A, u0, (0, 2.0), M)
        alg = ManifoldDiffEq.ManifoldLieEuler(M, ExponentialRetraction())
        sol1 = solve(prob, alg, dt = 1 / 8)
        @test alg_order(alg) == 1

        @test sol1(0.0) ≈ u0
    end

    @testset "Product manifold" begin
        A = ManifoldDiffEq.ManifoldDiffEqOperator{Float64}() do u, p, t
            return ProductRepr(cross(u.parts[1], [1.0, 0.0, 0.0]), u.parts[2])
        end
        u0 = ProductRepr([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
        M = ProductManifold(Sphere(2), Euclidean(3))
        prob = ManifoldDiffEq.ManifoldODEProblem(A, u0, (0, 2.0), M)
        alg = ManifoldDiffEq.ManifoldLieEuler(M, ExponentialRetraction())
        sol1 = solve(prob, alg, dt = 1 / 8)

        @test isapprox(M, sol1(0.0), u0)
    end


end
