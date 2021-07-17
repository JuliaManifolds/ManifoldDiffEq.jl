using Plots, Manifolds, ManifoldsBase, ManifoldDiffEq, Documenter, PyPlot
ENV["GKSwstype"] = "100"

makedocs(
    # for development, we disable prettyurls
    format=Documenter.HTML(prettyurls=false, assets=["assets/favicon.ico"]),
    modules=[ManifoldDiffEq],
    authors="Seth Axen, Mateusz Baran, Ronny Bergmann, and contributors.",
    sitename="ManifoldDiffEq.jl",
    pages=[
        "Home" => "index.md",
    ],
)
deploydocs(repo="github.com/JuliaManifolds/ManifoldDiffEq.jl.git", push_preview=true)
