function RecursiveArrayTools.recursive_unitless_bottom_eltype(
    ::Type{T},
) where {T<:ProductRepr}
    return recursive_unitless_bottom_eltype(T.parameters[1])
end
function RecursiveArrayTools.recursive_bottom_eltype(a::ProductRepr)
    return recursive_bottom_eltype(a.parts[1])
end
RecursiveArrayTools.recursive_unitless_eltype(::Type{T}) where {T<:ProductRepr} = T
RecursiveArrayTools.recursive_unitless_eltype(a::ProductRepr) = typeof(a)

function RecursiveArrayTools.recursivecopy!(x, y)
    map(copyto!, submanifold_components(x), submanifold_components(y))
    return x
end

function DiffEqBase.ODE_DEFAULT_NORM(u::ProductRepr, t)
    return sqrt(
        real(sum(p -> DiffEqBase.ODE_DEFAULT_NORM(p, t)^2, u.parts)) / length(u.parts),
    )
end
