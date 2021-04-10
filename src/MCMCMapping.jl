"""
    MCMCMapping
    
Functions used for mapping the connectivity of local minima of
a known Lipschitz function. The methods follow a stochastic
gradient descent trajectory and extract persistent homology
features in the augmented space of perturbation × parameters.

This work was produced as part of the project for the UIUC course
ECE 557, Geometric Control Theory.
"""
module MCMCMapping

using LinearAlgebra: norm

export generatePerturbation, MCMCTrajectory


"""
    generatePerturbation(samples, σ, windowlength)

Generates a random perturbation trajectory.

The trajectory is made of a random gaussian vector, filtered
through convolution with normalized Hann window of a given length.

### Arguments
 - `samples`      -- The number of timesteps for the perturbation
 - `σ`            -- The standard deviation of any given Gaussian sample
 - `windowlength` -- The length of the Hann window

 ### Returns
 A filtered Gaussian vector of length `samples`

 ### Notes 
 Performs the convolution directly, assumes ``windowlength << samples``
"""
function generatePerturbation(samples, σ, windowlength)

    perturbation = σ * randn(samples + windowlength)
    window = sin.((1:windowlength)/(windowlength+1) * π).^2

    # Rescale the window to preserve variance
    window ./= norm(window)

    return [sum(window .* perturbation[i:i+windowlength-1]) 
                                                    for i in 1:samples]
end



"""
    MCMCTrajectory(fₓ, x₀, α, stepsize, samples, σ, windowlength)

Generates a trajectory through the state × perturbation space based
on a noisy gradient descent process using the provided gradient.

The trajectory evolves according to ``x_{t+1} = x_t + α(uₜ - fₓ(x_t))``,
where ``uₜ`` represents the random perturbation.

### Arguments
 - `fₓ`           -- Gradient of the function of interest
 - `x₀`           -- Initial Point
 - `α`            -- Gradient Descent step size
 - `samples`      -- Number of timesteps for the trajectory
 - `σ`            -- Standard deviation of the Gaussian perturbation
 - `windowlength` -- Length of the Hann window for the perturbation process

### Returns
A trajectory in the state × perturbation space with states in the first
half of the rows, and perturbations in the second half.

### Notes 
Performs the convolution directly, assumes ``windowlength << samples``
"""
function MCMCTrajectory(fₓ, x₀, α, samples, σ, windowlength)
    N = length(x₀)
    
    # Initialize the state vector
    state = zeros(2*length(x₀),samples)
    state[1:N,1]     .= x₀
    @inbounds for i in 1:N
        state[N+i,:] .= generatePerturbation(samples,σ,windowlength)
    end

    @inbounds for i in 2:samples
        xₜ = @view state[1:N,i-1]
        uₜ = @view state[N+1:end,i]
        state[1:N,i] = xₜ + α * (uₜ .- fₓ(xₜ))
    end
    return state
end

end