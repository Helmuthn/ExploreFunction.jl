"""
    ExploreFunction
    
Functions used for mapping the connectivity of local minima of
a known Lipschitz function. The methods follow a stochastic
gradient descent trajectory and extract persistent homology
features in the augmented space of perturbation × parameters.
"""
module ExploreFunction

include("MCMCMapping.jl")
include("VietorisRips.jl")

end
