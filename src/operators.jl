"""
    AbstractVectorTransportOperator

Abstract type for vector transport operators in the frozen coefficients formulation.
"""
abstract type AbstractVectorTransportOperator end

struct DefaultVectorTransportOperator{TVT <: AbstractVectorTransportMethod} <:
    AbstractVectorTransportOperator
    vtm::TVT
end

"""
    (vto::DefaultVectorTransportOperator)(M::AbstractManifold, p, X, q, params, t_from, t_to)

In the frozen coefficient formulation, transport tangent vector `X` such that
`X = f(p, params, t_from)` to point `q` at time `t_to`. This provides a sort of estimation
of `f(q, params, t_to)`.
"""
function (vto::DefaultVectorTransportOperator)(
        M::AbstractManifold,
        p,
        X,
        q,
        params,
        t_from,
        t_to,
    )
    # default implementation, may be customized as needed.
    return vector_transport_to(M, p, X, q)
end

"""
    FrozenManifoldDiffEqOperator{T<:Number,TM<:AbstractManifold,TF,TVT} <: AbstractSciMLOperator{T}

DiffEq operator on manifolds in the frozen vector field formulation.
"""
struct FrozenManifoldDiffEqOperator{T <: Number, TF, TVT <: AbstractVectorTransportOperator} <:
    AbstractSciMLOperator{T}
    func::TF
    operator_vector_transport::TVT
end

function FrozenManifoldDiffEqOperator{T}(f, ovt) where {T <: Number}
    return FrozenManifoldDiffEqOperator{T, typeof(f), typeof(ovt)}(f, ovt)
end
function FrozenManifoldDiffEqOperator{T}(f) where {T <: Number}
    return FrozenManifoldDiffEqOperator{T}(
        f,
        DefaultVectorTransportOperator(ParallelTransport()),
    )
end

function (L::FrozenManifoldDiffEqOperator)(du, u, _u, p, t)
    return copyto!(du, L.func(u, p, t))
end
function (L::FrozenManifoldDiffEqOperator)(u, _u, p, t)
    return L.func(u, p, t)
end


"""
    LieManifoldDiffEqOperator{T<:Number,TF} <: AbstractSciMLOperator{T}

DiffEq operator on manifolds in the Lie group action formulation.
"""
struct LieManifoldDiffEqOperator{T <: Number, TF} <: AbstractSciMLOperator{T}
    func::TF
end

function LieManifoldDiffEqOperator{T}(f) where {T <: Number}
    return LieManifoldDiffEqOperator{T, typeof(f)}(f)
end

function (L::LieManifoldDiffEqOperator)(du, u, _u, p, t)
    return copyto!(du, L.func(u, p, t))
end
function (L::LieManifoldDiffEqOperator)(u, _u, p, t)
    return L.func(u, p, t)
end
