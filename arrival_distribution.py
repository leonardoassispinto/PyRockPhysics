#arraival_distribution.py

import pymc as pm
import numpy as np

def get_pymc_model(probabilities):
    def arrival_logp(value, probs):
        if value < 0 or value >= len(probs):
            return -np.inf
        prob = probs[value]
        if prob <= 0:
            return -np.inf
        else:
            return np.log(prob)

    def arrival_rand(probs):
        return np.random.choice(len(probs), p=probs)

    arrival_model = pm.Stochastic(
            logp = arrival_logp,
            doc = 'The index of the arrival.',
            name = 'arrival',
            parents = {'probs': probabilities},
            random = arrival_rand,
            trace = True,
            dtype=int,
            observed = False,
            cache_depth = 2,
            plot=True,
            verbose = 0)
    
    return arrival_model
