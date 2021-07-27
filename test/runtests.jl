using Test
using ManifoldDiffEq
using Manifolds
using OrdinaryDiffEq: OrdinaryDiffEq, alg_order
using LinearAlgebra
using DiffEqBase

function test_solver_frozen(manifold_to_alg; expected_order = nothing)
    expected_order !== nothing && @testset "alg_order" begin
        alg = manifold_to_alg(Sphere(2))
        @test alg_order(alg) == expected_order
    end

    @testset "Sphere" begin
        A = ManifoldDiffEq.ManifoldDiffEqOperator{Float64}() do u, p, t
            return cross(u, [1.0, 0.0, 0.0])
        end
        u0 = [0.0, 1.0, 0.0]
        M = Sphere(2)
        alg = manifold_to_alg(M)
        prob = ManifoldDiffEq.ManifoldODEProblem(A, u0, (0, 2.0), M)
        sol1 = solve(prob, alg, dt = 1 / 8)

        @test sol1(0.0) ≈ u0
    end

    @testset "Product manifold" begin
        A = ManifoldDiffEq.ManifoldDiffEqOperator{Float64}() do u, p, t
            return ProductRepr(cross(u.parts[1], [1.0, 0.0, 0.0]), u.parts[2])
        end
        u0 = ProductRepr([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
        M = ProductManifold(Sphere(2), Euclidean(3))
        alg = manifold_to_alg(M)
        prob = ManifoldDiffEq.ManifoldODEProblem(A, u0, (0, 2.0), M)
        sol1 = solve(prob, alg, dt = 1 / 8)

        @test isapprox(M, sol1(0.0), u0)
    end

end

function test_solver_lie(manifold_to_alg; expected_order = nothing)
    expected_order !== nothing && @testset "alg_order" begin
        action = RotationAction(Euclidean(3), SpecialOrthogonal(3))
        alg = manifold_to_alg(Sphere(2), action)
        @test alg_order(alg) == expected_order
    end

    @testset "Sphere" begin
        M = Sphere(2)
        action = RotationAction(Euclidean(3), SpecialOrthogonal(3))
        A = ManifoldDiffEq.ManifoldDiffEqOperator{Float64}() do u, p, t
            q = exp(M, u, cross(u, [1.0, 0.0, 0.0]))
            return optimal_alignment(action, u, q)
        end
        u0 = [0.0, 1.0, 0.0]
        alg = manifold_to_alg(M, action)
        prob = ManifoldDiffEq.ManifoldODEProblem(A, u0, (0, 2.0), M)
        sol1 = solve(prob, alg, dt = 1 / 8)

        @test sol1(0.0) ≈ u0
    end

end

@testset "ManifoldDiffEq" begin
    manifold_to_alg1 = M -> ManifoldDiffEq.ManifoldEuler(M, ExponentialRetraction())
    test_solver_frozen(manifold_to_alg1; expected_order = 1)

    manifold_to_alg2 = M -> ManifoldDiffEq.CG2(M, ExponentialRetraction())
    test_solver_frozen(manifold_to_alg2; expected_order = 2)

    manifold_to_alg3 =
        (M, action) -> ManifoldDiffEq.ManifoldLieEuler(M, ExponentialRetraction(), action)
    test_solver_lie(manifold_to_alg3; expected_order = 1)
end
