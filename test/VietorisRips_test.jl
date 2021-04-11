using ExploreFunction, Test

dataset = [ 0 0;
            0 1;
            0 3]

true_distance = [   0 1 3;
                    1 0 2;
                    3 2 0]
                    
@testset "euclideandistance" begin
    @test euclideandistance(dataset) == true_distance
end 


true_prune = [  0 3;
                3 0]
true_remaining = [1 3]

@testset "prunedataset" begin
    prune_dist, remaining = prunedataset(true_distance,1.5)
    @test prune_dist == true_prune 
    @test true_remaining == true_remaining
end

dataset_ext = vcat(dataset, [0 6])
true_distance_ext = [   0 1 3 6;
                        1 0 2 5;
                        3 2 0 3;
                        6 5 3 0]
@testset "H0Barcode" begin
    barcode = H0Barcode(true_distance_ext)
    @test barcode[2,:] == [1, 2, 3]
end


@testset "lowerneighbors" begin
    adjacency_test = zeros(4,4)
    adjacency_test[1,2] = adjacency_test[2,1] = 1
    adjacency_test[2,3] = adjacency_test[3,2] = 1
    adjacency_test[3,4] = adjacency_test[4,3] = 1
    adjacency_test[2,4] = adjacency_test[4,2] = 1

    n1 = ExploreFunction.lowerneighbors(1,adjacency_test)
    @test n1 == []

    n2 = ExploreFunction.lowerneighbors(2,adjacency_test)
    @test n2 == [1]

    n3 = ExploreFunction.lowerneighbors(3,adjacency_test)
    @test n3 == [2]

    n4 = ExploreFunction.lowerneighbors(4,adjacency_test)
    @test n4 == [2,3]
end

@testset "edgelist" begin
    adjacency_test = zeros(4,4)
    adjacency_test[1,2] = adjacency_test[2,1] = 1
    adjacency_test[2,3] = adjacency_test[3,2] = 1
    adjacency_test[3,4] = adjacency_test[4,3] = 1
    adjacency_test[2,4] = adjacency_test[4,2] = 1

    edges = ExploreFunction.edgelist(adjacency_test)
    true_edges = [  1 2 2 3;
                    2 3 4 4]
    @test true_edges == edges
end

@testset "neighborintersect" begin 
    adjacency_test = zeros(4,4)
    adjacency_test[1,2] = adjacency_test[2,1] = 1
    adjacency_test[2,3] = adjacency_test[3,2] = 1
    adjacency_test[3,4] = adjacency_test[4,3] = 1
    adjacency_test[2,4] = adjacency_test[4,2] = 1

    s1 = [3, 4]
    i1 = ExploreFunction.neighborintersect(s1,adjacency_test)
    @test [2] == i1

    s2 = [2, 3]
    i2 = ExploreFunction.neighborintersect(s2,adjacency_test)
    @test [] == i2
end

@testset "inductiveVR" begin 
    adjacency_test = zeros(4,4)
    adjacency_test[1,2] = adjacency_test[2,1] = 1
    adjacency_test[2,3] = adjacency_test[3,2] = 1
    adjacency_test[3,4] = adjacency_test[4,3] = 1
    adjacency_test[2,4] = adjacency_test[4,2] = 1

    VR = inductiveVR(adjacency_test,2)
    # Test the 1-simplices 
    true_s1 = [ 1 2 2 3;
                2 3 4 4]
    @test VR[1] == true_s1

    # Test the 2-simplices
    true_s2 = [2, 3, 4]
    @test VR[2][:] == true_s2

    # Test the 3-simplices 
    @test VR[3][:] == []
end