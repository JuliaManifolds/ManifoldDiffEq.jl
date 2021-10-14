using Test
using ManifoldDiffEq
using Manifolds
using OrdinaryDiffEq: OrdinaryDiffEq, alg_order
using LinearAlgebra
using DiffEqBase

function test_solver_frozen(manifold_to_alg; expected_order = nothing, adaptive = false)
    expected_order !== nothing && @testset "alg_order" begin
        alg = manifold_to_alg(Sphere(2))
        @test alg_order(alg) == expected_order
    end

    @testset "Sphere" begin
        M = Sphere(2)
        A = ManifoldDiffEq.FrozenManifoldDiffEqOperator{Float64}() do u, p, t
            return cross(u, [1.0, 0.0, 0.0])
        end
        u0 = [0.0, 1.0, 0.0]
        alg = manifold_to_alg(M)
        prob = ManifoldDiffEq.ManifoldODEProblem(A, u0, (0, 2.0), M)
        sol1 = if adaptive
            solve(prob, alg)
        else
            solve(prob, alg, dt = 1 / 8)
        end

        @test sol1(0.0) ≈ u0
        @test is_point(M, sol1(1.0))
    end

    @testset "Product manifold" begin
        M = ProductManifold(Sphere(2), Euclidean(3))
        A = ManifoldDiffEq.FrozenManifoldDiffEqOperator{Float64}() do u, p, t
            return ProductRepr(cross(u.parts[1], [1.0, 0.0, 0.0]), u.parts[2])
        end
        u0 = ProductRepr([0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
        alg = manifold_to_alg(M)
        prob = ManifoldDiffEq.ManifoldODEProblem(A, u0, (0, 2.0), M)
        sol1 = if adaptive
            solve(prob, alg)
        else
            solve(prob, alg, dt = 1 / 8)
        end

        @test isapprox(M, sol1(0.0), u0)
        @test is_point(M, sol1(1.0))
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

        A = ManifoldDiffEq.LieManifoldDiffEqOperator{Float64}() do u, p, t
            return hat(SpecialOrthogonal(3), Matrix(I(3)), cross(u, [1.0, 0.0, 0.0]))
        end
        u0 = [0.0, 1.0, 0.0]
        alg = manifold_to_alg(M, action)
        prob = ManifoldDiffEq.ManifoldODEProblem(A, u0, (0, 2.0), M)
        sol1 = solve(prob, alg, dt = 1 / 8)

        @test sol1(0.0) ≈ u0
        @test is_point(M, sol1(1.0))
    end

end

# constructing tableaus for comparison with DiffEq on the Euclidean space
function constructME(T::Type = Float64)
    A = fill(0, 1, 1)
    c = [0]
    α = [1]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    return (DiffEqBase.ExplicitRKTableau(A, c, α, 1))
end

function constructCG2(T::Type = Float64)
    A = [
        0 0
        1//2 0
    ]
    c = [0, 1 // 2]
    α = [0, 1]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    return (DiffEqBase.ExplicitRKTableau(A, c, α, 2))
end

function constructCG3(T::Type = Float64)
    A = [
        0 0 0
        3//4 0 0
        119//216 17/108 0
    ]
    c = [0, 3 // 4, 17 // 24]
    α = [13 // 51, -2 // 3, 24 // 17]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    return (DiffEqBase.ExplicitRKTableau(A, c, α, 3))
end

function constructRKMK4(T::Type = Float64)
    A = [
        0 0 0 0
        1//2 0 0 0
        0 1//2 0 0
        0 0 1 0
    ]
    c = [0, 1 // 2, 1 // 2, 1]
    α = [1 // 6, 1 // 3, 1 // 3, 1 // 6]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    return (DiffEqBase.ExplicitRKTableau(A, c, α, 3))
end

function compare_with_diffeq_frozen(manifold_to_alg, tableau)
    M = Euclidean(2)
    # damped harmonic oscillator d^2u/dt^2 + c du/dt + ku = 0
    k = -0.25
    c = 1
    f(u, p, t) = [-c * u[1] - k * u[2], u[1]]
    A = ManifoldDiffEq.FrozenManifoldDiffEqOperator{Float64}(f)
    u0 = [-1.0, 1.0]
    alg = manifold_to_alg(M)
    tspan = (0, 2.0)
    dt = 1 / 8
    prob_frozen = ManifoldDiffEq.ManifoldODEProblem(A, u0, tspan, M)
    sol_frozen = solve(prob_frozen, alg, dt = dt)

    alg_diffeq = OrdinaryDiffEq.ExplicitRK(tableau)
    prob_diffeq = ODEProblem(A, u0, tspan)
    sol_diffeq = solve(prob_diffeq, alg_diffeq; dt = dt, adaptive = false)

    @test isapprox(sol_frozen.u, sol_diffeq.u)
end

function compare_with_diffeq_lie(manifold_to_alg, tableau)
    M = Euclidean(2)
    # damped harmonic oscillator d^2u/dt^2 + c du/dt + ku = 0
    k = -0.25
    c = 1
    f(u, p, t) = [-c * u[1] - k * u[2], u[1]]
    action = TranslationAction(Euclidean(2), TranslationGroup(2))

    A = ManifoldDiffEq.LieManifoldDiffEqOperator{Float64}(f)
    u0 = [-1.0, 1.0]
    alg = manifold_to_alg(M, action)
    tspan = (0, 2.0)
    dt = 1 / 8
    prob_lie = ManifoldDiffEq.ManifoldODEProblem(A, u0, tspan, M)
    sol_lie = solve(prob_lie, alg, dt = dt)

    alg_diffeq = OrdinaryDiffEq.ExplicitRK(tableau)
    prob_diffeq = ODEProblem(A, u0, tspan)
    sol_diffeq = solve(prob_diffeq, alg_diffeq; dt = dt, adaptive = false)

    @test isapprox(sol_lie.u, sol_diffeq.u)
end


@testset "ManifoldDiffEq" begin
    manifold_to_alg_me = M -> ManifoldDiffEq.ManifoldEuler(M, ExponentialRetraction())
    test_solver_frozen(manifold_to_alg_me; expected_order = 1)
    compare_with_diffeq_frozen(manifold_to_alg_me, constructME())

    manifold_to_alg_cg2 = M -> ManifoldDiffEq.CG2(M, ExponentialRetraction())
    test_solver_frozen(manifold_to_alg_cg2; expected_order = 2)
    compare_with_diffeq_frozen(manifold_to_alg_cg2, constructCG2())

    manifold_to_alg_cg23 = M -> ManifoldDiffEq.CG23(M, ExponentialRetraction())
    test_solver_frozen(manifold_to_alg_cg23; expected_order = 2, adaptive = true)

    manifold_to_alg_cg3 = M -> ManifoldDiffEq.CG3(M, ExponentialRetraction())
    test_solver_frozen(manifold_to_alg_cg3; expected_order = 3)
    compare_with_diffeq_frozen(manifold_to_alg_cg3, constructCG3())

    manifold_to_alg_lie_euler =
        (M, action) -> ManifoldDiffEq.ManifoldLieEuler(M, ExponentialRetraction(), action)
    test_solver_lie(manifold_to_alg_lie_euler; expected_order = 1)
    compare_with_diffeq_lie(manifold_to_alg_lie_euler, constructME())

    manifold_to_alg_rkmk4 =
        (M, action) -> ManifoldDiffEq.RKMK4(M, ExponentialRetraction(), action)
    test_solver_lie(manifold_to_alg_rkmk4; expected_order = 4)
    compare_with_diffeq_lie(manifold_to_alg_rkmk4, constructRKMK4())
end
