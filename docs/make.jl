#!/usr/bin/env julia
#
#

using Plots, Manifolds, ManifoldsBase, ManifoldDiffEq, LieGroups, Documenter, PythonPlot
using DocumenterCitations, DocumenterInterLinks

ENV["GKSwstype"] = "100"

bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style = :alpha)
links = InterLinks(
    "Manifolds" => ("https://juliamanifolds.github.io/Manifolds.jl/stable/"),
    "ManifoldsBase" => ("https://juliamanifolds.github.io/ManifoldsBase.jl/stable/"),
    "LieGroups" => ("https://juliamanifolds.github.io/LieGroups.jl/stable/"),
)

makedocs(
    # for development, we disable prettyurls
    format = Documenter.HTML(
        prettyurls = (get(ENV, "CI", nothing) == "true"),
        assets = ["assets/favicon.ico", "assets/citations.css"],
    ),
    modules = [ManifoldDiffEq],
    authors = "Seth Axen, Mateusz Baran, Ronny Bergmann, and contributors.",
    sitename = "ManifoldDiffEq.jl",
    pages = [
        "Home" => "index.md",
        "Manifold solvers" => "manifold_solvers.md",
        "Lie group action solvers" => "lie_group_solvers.md",
        "Frozen coefficients solvers" => "frozen_coefficients.md",
        "Error estimation" => "error_estimation.md",
        "Notation" => "notation.md",
        "Examples" => "examples.md",
        "Internals" => "internals.md",
        "References" => "references.md",
    ],
    plugins = [bib, links],
)
deploydocs(
    repo = "github.com/JuliaManifolds/ManifoldDiffEq.jl.git",
    push_preview = true,
    devbranch = "main",
)
