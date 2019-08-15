using Documenter, HMCUtilities

makedocs(
    modules = [HMCUtilities],
    format = :html,
    checkdocs = :exports,
    sitename = "HMCUtilities.jl",
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/salilab/HMCUtilities.jl.git",
)
