#method.py

import numpy as np
import pymc as pm
import gamma_distribution
import prior_parameters
import arrival_distribution
from scipy.stats import boxcox

def infer_arrival(amplitude, time, arrival_prior_probs, fac, chainsize=50000, burnin=10000, thin=4, useboxcox=True):
    n = len(amplitude)

    dt = time[1] - time[0]
    t0 = time[0]
    
    # Box-Cox Transoformation to force datum normalization
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
    # https://en.wikipedia.org/wiki/Power_transform#Box-Cox_transformation
    if useboxcox:
        where = np.isfinite(amplitude)

        eps = 0.1
        c = -np.nanmin(amplitude[where]) + eps
        aux, lmbd = boxcox(amplitude[where]+c)
        if lmbd != 0.0:
            amplitude = ((amplitude + c)**lmbd - 1.0)/lmbd
        else:
            amplitude = np.log(amplitude + c)
    
    a1, b1, a2, b2 = prior_parameters.gammas(amplitude, arrival_prior_probs, fac, 1.0-fac)

    sigma2_1 = pm.InverseGamma("sigma2_1", a1, b1)
    sigma2_2 = pm.InverseGamma("sigma2_2", a2, b2)

    arrival = arrival_distribution.get_pymc_model(arrival_prior_probs)

    @pm.deterministic
    def precision(arrival=arrival, sigma2_1=sigma2_1, sigma2_2=sigma2_2):
        out = np.empty(n)
        out[:arrival] = sigma2_1
        out[arrival:] = sigma2_2
        return 1.0/out

    observation = pm.Normal("obs", 0.0, precision, value=amplitude, observed=True)
    model = pm.Model([observation, sigma2_1, sigma2_2, arrival], verbose=0)

    mcmc = pm.MCMC(model, verbose=0)
    mcmc.sample(chainsize, burnin, thin)

    arrival_samples = mcmc.trace('arrival')[:]

    return arrival_samples


def summarize(arrival_samples):
    mean = np.mean(arrival_samples)
    p05 = np.percentile(arrival_samples, 5)
    p95 = np.percentile(arrival_samples, 95)
    if p05 == p95:
        p05 = np.min(arrival_samples)
        p95 = np.max(arrival_samples)
    summary = {'map': mean, 'p05': p05, 'p95': p95}
    return summary
