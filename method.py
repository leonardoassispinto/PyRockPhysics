import numpy as np
import pymc as pm
import gamma_distribution
import prior_parameters
import arrival_distribution
from scipy.stats import boxcox

# TODO: dividir essa função em:
# 1. (DONE) Uma função (ou várias) que calcula a probabilidade a priori da primeira quebra ("arrival"): trapézio
#     1.1 Criar uma outra função que utiliza qualquer distribuição de velocidade, não somente de com thresholds (trapézio)
# 2. (DONE) Uma função (ou várias) que calcula os parâmetros da probabilidade a priori das variâncias (a1, b1, a2, b2) em função do dado e do retorno da função de cima
# 3. (DONE) Uma função que vai gerar as cadeias a partir dos resultados das duas funções acima
# 4. (DONE) Uma função (ou mais) que "sumariza" as cadeias (plota as cadeias, calcula MAP, etc)
# 5. Fazer com que a função infer_arrival já receba o dado com o tempo "pré-processado" (sem tempos negativos e descontado o face-to-face)

def infer_arrival(amplitude, time, arrival_prior_probs, fac, chainsize=50000, burnin=10000, thin=4, useboxcox=True):
    n = len(amplitude)

    dt = time[1] - time[0]
    t0 = time[0]
    
    # Transoformada de Box-Cox para forçar normalidade no dado
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
    summary = {'map': np.mean(arrival_samples)}
    return summary

