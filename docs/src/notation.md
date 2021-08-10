# Notation


Notation of ManifoldDiffEq.jl mostly follows the [notation of Manifolds.jl](https://juliamanifolds.github.io/Manifolds.jl/stable/misc/notation.html). There are, however, a few changes to more closely match the notation of the DiffEq ecosystem. Namely:

* `u` is often used to denote points on a manifold.
* Tangent vectors are usually denoted by ``X``, ``Y`` but some places may use the symbol ``k``.
* Parameters of the solved function are denoted either by ``p`` or `params`, depending on the context.
