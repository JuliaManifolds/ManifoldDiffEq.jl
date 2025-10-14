using Manifolds
using ManifoldDiffEq
using ManifoldsBase
import Manifolds.apply,
    Manifolds.apply_diff,
    Manifolds.apply_diff_group,
    Manifolds.base_group

using Random; Random.seed!(1)
using LinearAlgebra
using SparseArrays
using RecursiveArrayTools: ArrayPartition
using TensorToolbox
using OrdinaryDiffEq


using Plots
# include("MyManifoldIntegrator.jl")

case = "CP"
# case = "TT"

if case=="CP"
    r = 3
    n = 10
    V = (n, n, n)
    M = CPTensor(V, r)
    tspan = (0.0, 10.0)
    dt = 0.001
end

if case=="TT"
    S = (3, 3)
    n = 10
    V = (n, n, n)
    M = TTTensor(V, S)
    tspan = (0.0, 10.0)
    dt = 0.01
end

# using LieGroups: cross, GeneralLinearGroup, Identity
# G = foldl(cross, GeneralLinearGroup.(V))
# struct MyAction <: AbstractGroupAction{LeftAction} end


### Lie group DLRA ###

# using LieGroups
# Manifolds.Identity(::typeof(G)) = LieGroups.Identity(G)

# base_group(::MyAction) = G

# apply(::MyAction, g, p) = [g_i * p_i for (g_i, p_i) in zip(g.x, p)]
# apply_diff(::MyAction, g, X, p) = [g_i * X_i for (g_i, X_i) in zip(g.x, X)]
# apply_diff_group(::MyAction, ::Identity, X, p) = [X_i * p_i for (X_i, p_i) in zip(X.x, p)]

# function Manifolds.lie_bracket(G::typeof(G), X, Y)
#     return ArrayPartition(Tuple([X_i * Y_i - Y_i * X_i for (X_i, Y_i) in zip(X.x, Y.x)]))
# end

# Second derivative finite central differences
Delta = 10.0 / n
D1 = sparse(Tridiagonal(-ones(n - 1), zeros(n), ones(n - 1)) / Delta)
D2 = sparse(Tridiagonal(ones(n - 1), -2 * ones(n), ones(n - 1)) / Delta^2)
D2[1, :] .= 0; D2[end, :] .= 0
D2[:, 1] .= 0; D2[:, end] .= 0
O = D2

O = rand(n, n)
O = O - O'

F = FrozenManifoldDiffEqOperator{Float64}() do u, p, t
    return [O * u_i for u_i in u]
end

# Initial condition u(x, y, z) = 1 / (1 + x^2 + y^2 + z^2)
u0_full = zeros(n, n, n) # Initialize
for i in 1:n
    for j in 1:n
        for k in 1:n
            u0_full[i, j, k] =
                1 / (1 + (0.5 - i / n)^2 + (0.5 - j / n)^2 + (0.5 - k / n)^2) *
                # 1 / (1 + (i / n) * (j / n) * (k / n) / 2) *
                # sin((i / n) * (j / n) * (k / n) * pi) *
                # rand() *
                (i != 1) * (i != n) * (j != 1) * (j != n) * (k != 1) * (k != n) # Boundary conditions
        end
    end
end

# u0_full = full(randTTtensor(V, S))

# u0_full = full(randktensor([V...], 10))

if case=="CP"
    decomposition = cp_als(u0_full, r)
    u0 = [factor .* (decomposition.lambda').^(1 / 3) for factor in decomposition.fmat]
end

if case=="TT"
    decomposition = TTsvd(u0_full, reqrank=S)
    u0 = [tenmat(factor, 2) for factor in decomposition.cores]
end

@assert(is_point(M, u0))

problem = ManifoldODEProblem(F, u0, tspan, M)
algorithm = ManifoldDiffEq.ManifoldEuler(M, ExponentialRetraction()) # bodge
# algorithm = ManifoldEulerBackward(M, ExponentialRetraction()) # bodge
solution = solve(problem, algorithm, dt=dt)


### Full rank ODE ###


source = ones(V)
source[1, :, :] .= 0; source[end, :, :] .= 0;
source[:, 1, :] .= 0; source[:, end, :] .= 0;
source[:, :, 1] .= 0; source[:, :, end] .= 0;
function f(u, p, t)
    local tmp1 = similar(u)
    local tmp2 = similar(u)
    local tmp3 = similar(u)

    for a in 1:n
        for b in 1:n
            tmp1[:, a, b] = O * u[:, a, b]
        end
    end

    for a in 1:n
        for b in 1:n
            tmp2[a, :, b] = O * u[a, :, b]
        end
    end

    for a in 1:n
        for b in 1:n
            tmp3[a, b, :] = O * u[a, b, :]
        end
    end

    # return u .* (tmp1 + tmp2 + tmp3)
    return (tmp1 + tmp2 + tmp3) # bodge
    # return (tmp1 + tmp2 + tmp3) + source # bodge
end
prob = ODEProblem(f, u0_full, tspan)
sol = solve(prob)
# println("error when t = 1 is ", norm(full(ktensor(solution.u[end])) - sol.u[end]) / norm(sol.u[end]))


### Compare ###

# es = [norm(full(ktensor(solution.u[i])) - sol(t)) / norm(sol(t)) for (i, t) in enumerate(tspan[1]:dt:tspan[2])]

es = [norm(embed(M, solution.u[i]) - sol(t)) / norm(sol(t)) for (i, t) in enumerate(tspan[1]:dt:tspan[2])]

if case=="CP"
    bs = [norm(full(cp_als(sol(t), r)) - sol(t)) / norm(sol(t)) for (i, t) in enumerate(tspan[1]:dt:tspan[2])]
end

if case=="TT"
    bs = [norm(full(TTsvd(sol(t); reqrank=S)) - sol(t)) / norm(sol(t)) for (i, t) in enumerate(tspan[1]:dt:tspan[2])]
end

ts = [t for t in tspan[1]:dt:tspan[2]]
ns = [maximum(sol(t)) for t in ts]
# plot(yaxis=:log, ylims=[minimum([es..., ns...]), 1e0], yticks=[1e-3, 1e-2, 1e-1, 1e0], xlabel="time")
plot(yaxis=:log, xlabel="time")
plot!(ts, ns, label="norm")

# plot(solution.u[end][1][:, 1])
plot!(ts, es, label="error")
plot!(ts, bs, label="best possible error")
