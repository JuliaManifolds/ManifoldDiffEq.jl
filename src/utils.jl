function RecursiveArrayTools.recursivecopy!(x, y)
    map(copyto!, submanifold_components(x), submanifold_components(y))
    return x
end
