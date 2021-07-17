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
