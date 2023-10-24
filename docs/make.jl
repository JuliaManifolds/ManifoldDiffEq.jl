using Plots, Manifolds, ManifoldsBase, ManifoldDiffEq, Documenter, PythonPlot
using DocumenterCitations

ENV["GKSwstype"] = "100"

bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style = :alpha)

makedocs(
    # for development, we disable prettyurls
    format = Documenter.HTML(
        prettyurls = false,
        assets = ["assets/favicon.ico", "assets/citations.css"],
    ),
    modules = [ManifoldDiffEq],
    authors = "Seth Axen, Mateusz Baran, Ronny Bergmann, and contributors.",
    sitename = "ManifoldDiffEq.jl",
    pages = [
        "Home" => "index.md",
        "Lie group action solvers" => "lie_group_solvers.md",
        "Frozen coefficients solvers" => "frozen_coefficients.md",
        "Error estimation" => "error_estimation.md",
        "Notation" => "notation.md",
        "Examples" => "examples.md",
        "References" => "references.md",
    ],
    plugins = [bib],
)
deploydocs(
    repo = "github.com/JuliaManifolds/ManifoldDiffEq.jl.git",
    push_preview = true,
    devbranch = "main",
)
