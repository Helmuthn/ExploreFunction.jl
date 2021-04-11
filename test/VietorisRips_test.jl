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

