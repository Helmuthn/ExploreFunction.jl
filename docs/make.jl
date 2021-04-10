using ExploreFunction
using Documenter

DocMeta.setdocmeta!(ExploreFunction, :DocTestSetup, :(using ExploreFunction); recursive=true)

makedocs(;
    modules=[ExploreFunction],
    authors="Helmuth Naumer <hnaumer2@illinois.edu>",
    repo="https://github.com/Helmuthn/ExploreFunction.jl/blob/{commit}{path}#{line}",
    sitename="ExploreFunction.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
