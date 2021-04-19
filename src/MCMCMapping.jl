using LinearAlgebra: norm
using Random: MersenneTwister
using LoopVectorization: @avxt
using JuMP
using COSMO

export generatePerturbation, MCMCTrajectory, MinimumHoleSize


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
        state[1:N,i] = xₜ + α * (uₜ .- gradients[i-1,:])
    end
    gradients[:,end] =  fₓ(state[1:N,end])

    return (state, gradients)
end

MCMCTrajectory(fₓ, x₀, α, samples, σ, windowlength) = 
        MCMCTrajectory(defaultRNG, fₓ, x₀, α, samples, σ, windowlength)


"""
    GenerateConstraints(gradients)

Generates the constraints used to compute the minimum spacing between
the forward path and reverse path in the augmented space.

### Arguments
 - `gradients`  -- Gradients along the trajectory
 
### Returns
A matrix `A` and vector `b` such that min_x ||Ax - b|| s.t. x≥0 results in the 
minimum path discrepancy using the threshold approximation.
"""
function GenerateConstraints(gradients)
    return (hcat(gradients',-gradients'), 2*sum(abs2,gradients,dims=1))
end

"""
    SolveMinProb(A,b)

Helper function solving minₓ ||Ax - b|| s.t. x≥0 giving the minimum size of the
hole in the augmented space. Uses JuMP and COSMO for now, could be changed later
"""
function SolveMinProb(A,b)
    M, N = size(A)

    model = Model(COSMO.Optimizer)
    set_silent(model)
    @variable(model, x[1:N] >= 0)
    @objective(model, Min, x' * A' * A * x - 2 * b' * A * x)
    optimize!(model)

    return objective_value(model)
end

"""
    MinimumHoleSize(gradients)

Given a matrix representing the gradients along a path, 
returns the minimum size of the geometric hole for trajectories
along that path.

### Arguments
 - `gradients` -- M×N matrix of N timesteps of gradients

### Returns 
The list of the gradeints
"""
function MinimumHoleSize(gradients)
    A, b = GenerateConstraints(gradients)
    return SolveMinProb(A, b)
end