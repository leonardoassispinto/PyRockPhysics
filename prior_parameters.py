import pymc as pm
import gamma_distribution
import numpy as np

def tau_trap(time, L, vmin_hard, vmin_soft, vmax_hard, vmax_soft):
    tmax_hard = L/vmin_hard
    tmax_soft = L/vmin_soft
    tmin_hard = L/vmax_hard
    tmin_soft = L/vmax_soft

    r1 = (time >= tmin_hard) & (time < tmin_soft)
    r2 = (time >= tmin_soft) & (time < tmax_soft)
    r3 = (time >= tmax_soft) & (time < tmax_hard)
    
    trap = np.zeros_like(time)
    trap[r1] = (time[r1] - tmin_hard)/(tmin_soft - tmin_hard)
    trap[r2] = 1.0
    trap[r3] = (tmax_hard - time[r3])/(tmax_hard - tmax_soft)

    trap /= np.sum(trap)

    return trap

def gammas(amplitude, tau_prior, fac_noise, fac_signal):
    tau_cdf = np.cumsum(tau_prior)

    r_noise = tau_cdf <= fac_noise
    r_signal = tau_cdf >= fac_signal

    a_noise = gamma_distribution.estimate_a(amplitude[r_noise])
    b_noise = gamma_distribution.estimate_b(amplitude[r_noise])
    a_signal = gamma_distribution.estimate_a(amplitude[r_signal])
    b_signal = gamma_distribution.estimate_b(amplitude[r_signal])

    return a_noise, b_noise, a_signal, b_signal
