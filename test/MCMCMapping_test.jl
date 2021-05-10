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

@testset "minimumdistance" begin
    A = zeros(1,3)
    B = zeros(1,4)
    A[1,:] = [1; 3; 4]
    B[1,:] = [5; 6; 7; 8]
    @test minimumdistance(A,B) == 1
    @test minimumdistance(B,A) == 1
    @test minimumdistance(A,A) == 0
    B[1,:] = [9; 6; 7; 8]
    @test minimumdistance(A,B) == 2
end


@testset "detectTransitions" begin
    disc_test = zeros(100)
    threshold = 0.5
    pathlength = 21
    disc_test[10] = 1
    disc_test[30] = 1

    unstable, stable = detectTransitions(disc_test,threshold,pathlength)

    @test unstable == [20, 40]
    stable_true = zeros(Int,98)

    for i in 1:98
        if i<10
            stable_true[i] = i + 10
        elseif i<29
            stable_true[i] = i + 11
        else
            stable_true[i] = i + 12
        end
    end
    @test stable == stable_true
end