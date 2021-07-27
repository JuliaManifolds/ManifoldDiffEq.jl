
abstract type AbstractVectorTransportOperator end

struct DefaultVectorTransportOperator{
    TM<:AbstractManifold,
    TVT<:AbstractVectorTransportMethod,
} <: AbstractVectorTransportOperator
    M::TM
    vtm::TVT
end

"""
    (vto::DefaultVectorTransportOperator)(p, X, q, params, t)

In the frozen coefficient formulation, transport tangent vector `X` such that
`X = f(p, params, t)` to point `q`. for a given 
"""
function (vto::DefaultVectorTransportOperator)(p, X, q, params, t)
    # default implementation, may be customized as needed.
    return vector_transport_to(vto.M, p, X, q)
end

"""
    FrozenManifoldDiffEqOperator{T<:Number,TM<:AbstractManifold,TF,TVT} <: SciMLBase.AbstractDiffEqOperator{T}

DiffEq operator on manifolds in the frozen vector field formulation
"""
struct FrozenManifoldDiffEqOperator{T<:Number,TF,TVT<:AbstractVectorTransportOperator} <:
       SciMLBase.AbstractDiffEqOperator{T}
    func::TF
    operator_vector_transport::TVT
end

function FrozenManifoldDiffEqOperator{T}(f, ovt) where {T<:Number}
    return FrozenManifoldDiffEqOperator{T,typeof(f),typeof(ovt)}(f, ovt)
end
function FrozenManifoldDiffEqOperator{T}(f, M::AbstractManifold) where {T<:Number}
    return FrozenManifoldDiffEqOperator{T}(
        f,
        DefaultVectorTransportOperator(M, ParallelTransport()),
    )
end

function (L::FrozenManifoldDiffEqOperator)(du, u, p, t)
    return copyto!(du, L.func(u, p, t))
end
function (L::FrozenManifoldDiffEqOperator)(u, p, t)
    return L.func(u, p, t)
end


"""
    ManifoldDiffEqOperator{T<:Number,TF} <: AbstractDiffEqOperator{T}

DiffEq operator on manifolds.
"""
struct ManifoldDiffEqOperator{T<:Number,TF} <: SciMLBase.AbstractDiffEqOperator{T}
    func::TF
end

ManifoldDiffEqOperator{T}(f) where {T<:Number} = ManifoldDiffEqOperator{T,typeof(f)}(f)

function (L::ManifoldDiffEqOperator)(du, u, p, t)
    return copyto!(du, L.func(u, p, t))
end
function (L::ManifoldDiffEqOperator)(u, p, t)
    return L.func(u, p, t)
end
