using ExploreFunction
using SafeTestsets

@safetestset "MCMCMapping.jl" begin
    include("MCMCMapping_test.jl")
end

@safetestset "VietorisRips.jl" begin
    include("VietorisRips_test.jl")
end