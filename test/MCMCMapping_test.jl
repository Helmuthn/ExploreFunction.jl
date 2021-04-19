using ExploreFunction, Test, Random

# Only basic tests for now


@testset "generatePerturbation" begin
    @testset "Power" begin
        rng = MersenneTwister(1234)
        perturb = generatePerturbation(rng,1000000,10,1)
        @test 10 ≈ sqrt(sum(abs2,perturb)/1000000) atol=0.1
    end 
    @testset "length" begin
        perturb = generatePerturbation(100,1,1)
        @test length(perturb) == 100
    end
end


# Should add more tests here at some point
@testset "MCMCTrajectory" begin
    @testset "Dimensions" begin
        x₀ = [1, 2]
        fₓ(x) = x
        traject, ~ = MCMCTrajectory(fₓ,x₀,1,100,1,10)
        @test size(traject) == (4,100)
    end
end