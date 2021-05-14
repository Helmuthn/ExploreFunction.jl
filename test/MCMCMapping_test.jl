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


@testset "GenerateConstraints" begin
    directions = randn(3,5)
    gradients = randn(3,5)

    A1_true = directions[:,1:2]
    A2_true = -directions[:,4:5]
    b1_true = zeros(2)
    b2_true = zeros(2)
    b1_true[1] = A1_true[:,1]' * gradients[:,1]
    b1_true[2] = A1_true[:,2]' * gradients[:,2]
    b2_true[1] = A2_true[:,1]' * gradients[:,4]
    b2_true[2] = A2_true[:,2]' * gradients[:,5]

    A1, A2, b1, b2 = ExploreFunction.GenerateConstraints(gradients, directions)
    @test A1 == A1_true'
    @test A2 == A2_true'
    @test b1 == b1_true 
    @test b2 == b2_true
end

@testset "SolveMinProg" begin
    A1 = [1/sqrt(2)  1/sqrt(2);
          1/sqrt(2) -1/sqrt(2)]
    A2 = -A1;
    b1 = ones(2)
    b2 = ones(2)
    true_min_dist = 2sqrt(2)
    @test true_min_dist ≈ ExploreFunction.SolveMinProb(A1,A2,b1,b2, 0, false) atol=0.01
end

@testset "MinimumHoleSize" begin
    gradients = ones(1,5)
    gradients[1,3] = 0
    gradients[1,4:5] .= -1
    directions = ones(1,5) 
    
    true_size = 2
    @test true_size ≈ MinimumHoleSize(gradients,directions, 0) atol=0.01
end

@testset "FormNetwork" begin
    dataset = 0.01randn(2,90)
    dataset[1,11:20] += range(0,1,length=10)
    dataset[1,21:end] .+= 1
    dataset[2,31:40] += range(0,1,length=10)
    dataset[2,41:end] .+= 1
    dataset[1,51:60] -= range(0,1,length=10)
    dataset[1,61:end] .-= 1
    dataset[2,71:80] -= range(0,1,length=10)
    dataset[2,81:end] .-= 1

    transitions = zeros(40)
    transitions[1:10] = 11:20
    transitions[11:20] = 31:40
    transitions[21:30] = 51:60
    transitions[31:40] = 71:80

    threshold = 0.2

    edges, mean = FormNetwork(dataset,transitions,threshold)
    # Sort the edges lexicographical order
    sort!(edges,by= x -> 10*minimum(x) + maximum(x))
    edges_comp = zeros(Int,2,length(edges))
    for i in 1:length(edges)
        if edges[i][1] < edges[i][2]
            edges_comp[1,i] = edges[i][1]
            edges_comp[2,i] = edges[i][2]
        else
            edges_comp[2,i] = edges[i][1]
            edges_comp[1,i] = edges[i][2]
        end
    end
    
    true_means = zeros(size(mean))
    true_means[:,1] = [0,0]
    true_means[:,2] = [1,0]
    true_means[:,3] = [1,1]
    true_means[:,4] = [0,1]

    true_edges = [  1 1 2 3;
                    2 4 3 4]

    @test mean ≈ true_means atol = 0.1
    @test true_edges == edges_comp
end