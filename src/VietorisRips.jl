using LinearAlgebra: I, norm
# using CairoMakie

export euclideandistance, inductiveVR, edgelist, H0Barcode, prunedataset

"""
euclideandistance(dataset)

Computes the Euclidean distance matrix for a set of points.
This corresponds to the weighted neighborhood graph when
the maximum allowable distance is greater than the maximum 
distance between points.

### Arguments
 - `dataset` -- An N×M array containing N points in Rᴹ

### Returns
An N×N symmetric matrix of pairwise distances between points.
"""
function euclideandistance(dataset)
    N = size(dataset)[1]
    distances = zeros(N,N)
    for i in 1:N
        for j in i+1:N
            distances[i,j] = norm(dataset[i,:] - dataset[j,:])
            distances[j,i] = distances[i,j]
        end
    end
    return distances
end

"""
lowerneighbors(k, distances)

Returns a list of the neighboring nodes which are lower in ordering.

### Arguments
 - `k`         -- Chosen node
 - `distances` -- Matrix with zeros when nodes are not connected

### Returns
An array of node neighbors with ``index<k``
"""
function lowerneighbors(k, distances)
    return findall(!iszero, distances[k,1:k-1])
end


function edgelist(distances)
    N = size(distances)[1]
    simplices = zeros(Int,2,N^2)
    ind = 1
    for i in 1:N-1
        for j in i+1:N
            if !iszero(distances[i,j])
                simplices[1, ind] = i
                simplices[2, ind] = j
                ind += 1
            end
        end
    end
    return simplices[:,1:ind-1]
end

function neighborintersect(simplex,distances)
    N = size(distances)[1]
    neighbors = [l for l in 1:N]
    for l in 1:size(simplex)[1]
        intersect!(neighbors,lowerneighbors(simplex[l],distances))
    end
    return neighbors
end

"""
    inductiveVR(distances, k)

Computes the Vietoris-Rips complex using the inductive algorithm.
Builds the simplices starting from 1-simplices and forming higher
dimensional faces.

### Arguments
 - `distances` -- Matrix with zeros when nodes are not connected
 - `k`         -- Maximum simplice dimension

### Returns
An array of simplicial matrices up to dimension `k`

### Notes
This is the most basic construction mentioned in the paper.

### Reference

Afra Zomorodian,
Fast construction of the Vietoris-Rips complex,
Computers & Graphics,
Volume 34, Issue 3,
2010,
Pages 263-271,
ISSN 0097-8493,
https://doi.org/10.1016/j.cag.2010.03.007.
"""
function inductiveVR(distances, k)
    N = size(distances)[1]

    # 1-Simplices
    V = [edgelist(distances)]

    # Iteratively expand the complex
    
    for i in 1:k
        simplices = zeros(Int,i+2,0)

        # For each i-simplex
        Threads.@threads for j in 1:size(V[end])[2]
            simplex = V[end][:,j]

            # Compute interaction of neighbors
            neighbors = neighborintersect(simplex,distances)

            # Append each element of the intersection into the simplices
            for v in neighbors
                simplices = hcat(simplices, sort([v, simplex...]))
            end
        end
        
        # Append the next level of simplices.
        push!(V,simplices)
    end
    return V
end

"""
    H0Barcode(distances)

Compute the H0 persistent homology from a euclidean distance matrix

### Arguments
 - `distances` -- A Euclidean Distance Matrix

### Returns
Sequence of H0 homology lifetimes
"""
function H0Barcode(distances)

    # Initialize the arrays
    n = size(distances)[1]
    persistence = zeros(2,n)
    dist_sort = zeros(n*(n-1) ÷ 2)
    indices = zeros(Int,2,n*(n-1) ÷ 2)
    rips_complex = Array(1:n)

    # Sort the distances for Birth-Death Process
    p = sortperm(distances[:])[n+2:2:length(distances)]
    i = 1
    @inbounds for ind in p
        a, b = divrem(ind,n)
        indices[1,i] = a+1
        indices[2,i] = b+1
        dist_sort[i] = distances[ind]
        i += 1
    end


    # Step through the sorted distances to record death process
    @inbounds for i in 1:length(dist_sort)
        # remove later homology class, copy into first
        hom1 = rips_complex[indices[1,i]]
        hom2 = rips_complex[indices[2,i]]

        if hom1 != hom2
            persistence[2,hom2] = dist_sort[i]
            for j in 1:n
                if rips_complex[j] == hom2
                    rips_complex[j] = hom1
                end
            end
        end
    end

    return persistence[:,2:end]
end

# function plotH0Barcode(H0)
#     fig = Figure(   resolution = (1200, 700), 
#                 backgroundcolor = RGBf0(0.98, 0.98, 0.98),
#                 fontsize=32)
#     ax1 = fig[1, 1] = Axis(fig, title = "H0 Persistent Homology")
#     for i in 1:size(H0)[2]
#         lines!(H0[:,i],[i,i])
#     end
#     n_ind = size(H0)[2]+1
#     lines!([0,H0[2,end]*1.3],[n_ind,n_ind])
#     lines!([H0[2,end]*1.3,H0[2,end]*1.4],[n_ind,n_ind],linestyle=:dash)
#     ax1.xlabel = "Rips Connection Distance"
#     return fig, ax1
# end

function sortH0deaths!(indices,dist_sort,distances)
    ind = 0
    @inbounds for i in 1:n-1 
        for j in i+1:n
            ind += 1
            indices[1,ind] = i
            indices[2,ind] = j
            dist_sort[ind] = distances[i,j]
        end
    end
end

"""
    prunedataset(distances, ϵ)

Prunes the dataset based on a Euclidean distance matrix.
Given a point, by the triangle inequality, any two points within
distance ``ϵ`` of the first point are within ``2ϵ`` of each other.

Beginning with the first element of the dataset, iterate up removing
all later elements within ``ϵ``. 
Should only change the persistent homology by up to ``ϵ``

### Arguments 
 - `distances`  -- Euclidean Distance Matrix 
 - `ϵ`          -- Pruning Distance
### Returns
Two values, `pruned_dist, remaining_indices`

 - `pruned_dist`        -- Distance matrix of remaining indices
 - `remaining_indices`  -- Remaining indices
"""
function prunedataset(distances, ϵ)
    N = size(distances)[1]

    # Find initial connections
    removals = 1 .+ findall(distances[1,2:end] .< ϵ)
    
    # For each of the later nodes
    for i in 2:N

        # If the node hasn't been removed yet, remove all close nodes
        if !(i in removals)
            union!(removals, i .+ findall(distances[i,i+1:end] .< ϵ))
        end
    end

    remaining_indices = setdiff(1:N,removals)
    pruned_dist = distances[remaining_indices,remaining_indices]

    return pruned_dist, remaining_indices
end