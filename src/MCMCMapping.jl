using LinearAlgebra: norm
using Random: MersenneTwister
using LoopVectorization: @avxt
using JuMP
using OSQP

export generatePerturbation, MCMCTrajectory, MinimumHoleSize, detectTransitions, minimumdistance, FormNetwork, GenerateConstraints


defaultRNG = MersenneTwister()

"""
    generatePerturbation([ rng,] samples, σ, windowlength)

Generates a random perturbation trajectory.

The trajectory is made of a random gaussian vector, filtered
through convolution with normalized Hann window of a given length.

### Arguments
 - `rng`          -- Random number generator (Optional)
 - `samples`      -- The number of timesteps for the perturbation
 - `σ`            -- The standard deviation of any given Gaussian sample
 - `windowlength` -- The length of the Hann window

 ### Returns
 A filtered Gaussian vector of length `samples`

 ### Notes 
 Performs the convolution directly, assumes ``windowlength << samples``
"""
function generatePerturbation(rng, samples, σ, windowlength)

    perturbation = σ * randn(rng, samples + windowlength)
    window = sin.((1:windowlength)/(windowlength+1) * π).^2

    # Rescale the window to preserve variance
    window ./= norm(window)

    return [sum(window .* perturbation[i:i+windowlength-1]) 
                                                    for i in 1:samples]
end

export generatePerturbation_v
function generatePerturbation_v(rng,samples, σ, windowlength)

    perturbation = σ * randn(rng, samples + windowlength)
    window = sin.((1:windowlength)/(windowlength+1) * π).^2

    # Rescale the window to preserve variance
    window ./= norm(window)

    # Allocate Memory
    out = zeros(samples)
    @avxt for i in 1:samples 
        for j in 1:windowlength
            out[i] += window[j] * perturbation[i+j-1]
        end
    end

    return out
end

generatePerturbation(samples, σ, windowlength) = 
        generatePerturbation(defaultRNG, samples, σ, windowlength)

generatePerturbation_v(samples, σ, windowlength) = 
        generatePerturbation_v(defaultRNG, samples, σ, windowlength)


"""
    MCMCTrajectory([rng,] fₓ, x₀, α, stepsize, samples, σ, windowlength)

Generates a trajectory through the state × perturbation space based
on a noisy gradient descent process using the provided gradient.

The trajectory evolves according to ``x_{t+1} = x_t + α(uₜ - fₓ(x_t))``,
where ``uₜ`` represents the random perturbation.

### Arguments
 - `rng`          -- Random number generator (Optional)
 - `fₓ`           -- Gradient of the function of interest
 - `x₀`           -- Initial Point
 - `α`            -- Gradient Descent step size
 - `samples`      -- Number of timesteps for the trajectory
 - `σ`            -- Standard deviation of the Gaussian perturbation
 - `windowlength` -- Length of the Hann window for the perturbation process

### Returns
A trajectory in the state × perturbation space with states in the first
half of the rows, and perturbations in the second half.
Additionally, returns the gradients along the path.

### Notes 
Performs the convolution directly, assumes ``windowlength << samples``
"""
function MCMCTrajectory(rng, fₓ, x₀, α, samples, σ, windowlength)
    N = length(x₀)
    
    # Initialize the state vector
    state = zeros(2*length(x₀),samples)
    state[1:N,1]     .= x₀
    @inbounds for i in 1:N
        state[N+i,:] .= generatePerturbation_v(rng, samples,σ,windowlength)
    end
    gradients = zeros(N, samples)

    @inbounds for i in 2:samples
        xₜ = @view state[1:N,i-1]
        uₜ = @view state[N+1:end,i]
        gradients[:,i-1] = fₓ(xₜ)
        state[1:N,i] = xₜ + α * (uₜ .- gradients[:,i-1])
    end
    gradients[:,end] =  fₓ(state[1:N,end])

    return (state, gradients)
end

MCMCTrajectory(fₓ, x₀, α, samples, σ, windowlength) = 
        MCMCTrajectory(defaultRNG, fₓ, x₀, α, samples, σ, windowlength)


"""
    resampleSpatialTrajectory(state, δ)

Resamples the trajectory by removing steps closer than δ in parameters.
"""
function resampleSpatialTrajectory(state, δ)
    M, N = size(state)
    δ = δ^2
    param_range = 1:Int(M/2)
    keep = ones(Bool,N)
    lastind = 1
    for i in 2:N
        if sum(abs2,state[param_range,i] - state[param_range,lastind]) < δ
            keep[i] = false
        else
            lastind = i
        end
    end
    return keep
end


"""
    GenerateConstraints(gradients, directions)

Generates the constraints used to compute the minimum spacing between
the forward path and reverse path in the augmented space.

### Arguments
 - `gradients`  -- Gradients along the trajectory
 
### Returns
A matrix `A` and vector `b` such that min_x ||Ax - b|| s.t. x≥0 results in the 
minimum path discrepancy using the threshold approximation.
"""
function GenerateConstraints(gradients,directions)

    # Split into two halves
    M, N = size(gradients)
    mid = Int(ceil(N/2))

    A1 = directions[:,1:mid-1]
    A2 = -directions[:,mid+1:end]

    b1 = zeros(mid-1)
    b2 = zeros(mid-1)
    @inbounds @simd for i in 1:mid-1
        b1[i] = directions[:,i]' * gradients[:,i]
        b2[i] = -directions[:,mid+i]' * gradients[:,mid+i]
    end

    return (A1', A2', b1, b2)
end

"""
    SolveMinProb(A1, A2, b1, b2, λ)

Helper function solving minₓ ||Ax - b|| s.t. x≥0 giving the minimum size of the
hole in the augmented space. Uses JuMP and OSQP for now, could be changed later
"""
function SolveMinProb(A1, A2, b1, b2, λ)
    M1, N1 = size(A1)
    M2, N2 = size(A2)

    # Flag for monotonic --> no constraints = zero distance
    if M1 == 0 || M2 == 0
        return 0
    end

    model = Model(OSQP.Optimizer)
    set_silent(model)
    @variable(model, x[1:N1])
    @variable(model, y[1:N2])
    @constraint(model, A1 * x .>= b1)
    @constraint(model, A2 * y .>= b2)
    @objective(model, Min, x' * x - 2 * y' * x + y' * y + λ*(x' * x + y' * y))
    optimize!(model)

    x_v = value.(x)
    y_v = value.(y)

    v_comp = x_v' * x_v - 2 * y_v' * x_v + y_v' * y_v

    if isinf(v_comp) || v_comp < 0
        return 0
    else
        return sqrt(v_comp)
    end
end

"""
    MinimumHoleSize(gradients, directions, λ)

Given a matrix representing the gradients along a path, 
returns the minimum size of the geometric hole for trajectories
along that path.

### Arguments
 - `gradients`  -- M×N matrix of N timesteps of gradients
 - `directions` -- M×N matrix of N timesteps of step directions

### Returns 
The list of the gradeints
"""
function MinimumHoleSize(gradients, directions, λ)
    A1, A2, b1, b2 = GenerateConstraints(gradients, directions)
    return SolveMinProb(A1, A2, b1, b2, λ)
end

"""
    detectTransitions(discrepancies, threshold, pathlength)

Detects the transitions by applying a simple threshold
"""
function detectTransitions(discrepancies, threshold, pathlength)
    unstable = Int.(findall(discrepancies .> threshold) .+ floor(pathlength/2))
    stable = Int.(findall(discrepancies .<= threshold) .+ floor(pathlength/2))
    return unstable, stable
end

"""
    minimumdistance(A,B)

Returns the minimum distance between two sets of points.
Each column corresponds to a point.
"""
function minimumdistance(A,B)
    M1, N1 = size(A)
    M2, N2 = size(B)

    min_dist = Inf * ones(N1)

    Threads.@threads for i in 1:N1
        @inbounds for j in 1:N2
            min_dist[i] = min(min_dist[i],sum(abs2,A[:,i] - B[:,j]))
        end
    end

    return sqrt(minimum(min_dist))
end

"""
    FormNetwork(dataset, holesizes)

Takes a dataset of positions in the parameter space 
and the transition labels, generates the
network structure.

Uses a naive approach of sorting the points into clusters.
"""
function FormNetwork(dataset, transitions, threshold)

    # Begin by splitting the dataset along the transitions
    # These form the initial clusters.
    clusterindices = [Array{Int}(1:transitions[1]-1)]
    last_ind = transitions[1]
    for i in 2:length(transitions)
        if transitions[i] > last_ind+2
            push!(clusterindices, last_ind+1:transitions[i]-1)
        end
        last_ind = transitions[i]
    end
    if transitions[end] < size(dataset)[2]
        push!(clusterindices, transitions[end]+1:size(dataset)[2])
    end

    # Next, cluster the clusters by a threshold, recording indices
    clusters_included = [[i] for i in 1:length(clusterindices)]
    ind1 = 1
    while ind1 < length(clusterindices)
        ind2 = ind1+1
        no_merge = true
        while ind2 <= length(clusterindices)
            cluster1 = @view dataset[:,clusterindices[ind1]]
            cluster2 = @view dataset[:,clusterindices[ind2]]
            dist = minimumdistance(cluster2,cluster1)

            # If clusters are close enough, merge
            if dist < threshold
                append!(clusterindices[ind1], clusterindices[ind2])
                append!(clusters_included[ind1],clusters_included[ind2])
                deleteat!(clusterindices,ind2)
                deleteat!(clusters_included,ind2)
                no_merge = false
            else
                ind2 += 1
            end
        end
        if no_merge
            ind1 += 1
        end
    end

    # Using sequential connections, form the graph (assuming sparsity)
    edges = []
    for i in 1:length(clusters_included)
        for j in i+1:length(clusters_included)
            # Check for inclusion of sequential initial clusters
            offsets = [i1 - i2  for i1 in clusters_included[i] 
                                for i2 in clusters_included[j]]
            connection = findfirst((x)-> (abs(x)==1), offsets)
            if !isnothing(connection)
                push!(edges, [i,j])
            end
        end
    end

    # Finally, as a coarse approximation, compute the means of the clusters
    means = zeros(2,length(clusterindices))
    for i in 1:length(clusterindices)
        means[:,i] = sum(dataset[:,clusterindices[i]], dims=2)/length(clusterindices[i])
    end

    return (edges, means)
end