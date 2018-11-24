#firstbreak.py

import arrival_distribution as ar_dis
import gamma_distribution as gamma
import method as met
import picoscopereader as psdata
import prior_parameters as prior
import numpy as np
import sys
import matplotlib.pyplot as plt
import plug_parameter as plug
import os

def first_break(time, wave, t0=0.0, nchains=5, ndiscard=1, length=None, vmin=None, vmax=None, run_twice=False):
    time -= t0
    dt = time[1] - time[0]

    where = (time >= 0.0)
    
    if length is None or vmin is None or vmax is None:
        arrival_prior_probs = np.zeros_like(time)
        arrival_prior_probs[where] = 1.0
        arrival_prior_probs /= np.sum(arrival_prior_probs)
        factor = 0.05
    else:
        vmin_hard = vmin - 0.2*(vmax - vmin)
        vmax_hard = vmax + 0.2*(vmax - vmin)
        arrival_prior_probs = met.prior_parameters.tau_trap(time, length, vmin_hard, vmin, vmax_hard, vmax)
        factor = 0.01

    tas = []
    p05s = []
    p95s = []
    for j in range(nchains):
        arrival_samples = met.infer_arrival(wave[where], time[where], arrival_prior_probs[where], factor, chainsize=10000, burnin=2000, thin=4, useboxcox=False)
        summary = met.summarize(arrival_samples)
        arrival = summary['map']
        p05 = summary['p05']
        p95 = summary['p95']

        tas.append(arrival*dt + time[where][0])
        p05s.append(p05*dt + time[where][0])
        p95s.append(p95*dt + time[where][0])

    tas = np.array(tas)
    p05s = np.array(p05s)
    p95s = np.array(p95s)

    keep_indexes = np.argsort(tas)[ndiscard:-ndiscard]
    
    ta_1 = tas[keep_indexes].mean()
    p05_1 = p05s[keep_indexes].mean()
    p95_1 = p95s[keep_indexes].mean()

    if not run_twice or (length is None or vmin is None or vmax is None):
        return {"arrival": ta_1, "p05": p05_1, "p95": p95_1}

    position = np.argmin(np.abs(ta_1 - time))
    window_size = 200
    p_start = max(0, position-window_size)
    p_end = min(len(time), position+window_size)
    new_time = time[p_start:p_end]
    new_wave = wave[p_start:p_end]

    where = (new_time >= 0.0)
    arrival_prior_probs = met.prior_parameters.tau_trap(new_time, length, vmin_hard, vmin, vmax_hard, vmax)
    
    tas = []
    p05s = []
    p95s = []
    for k in range(nchains):
        arrival_samples = met.infer_arrival(new_wave[where], new_time[where], arrival_prior_probs[where], 0.01, chainsize=10000, burnin=2000, thin=4, useboxcox=False)
        summary = met.summarize(arrival_samples)
        arrival = summary['map']
        p05 = summary['p05']
        p95 = summary['p95']

        tas.append(arrival*dt + new_time[where][0])
        p05s.append(p05*dt + new_time[where][0])
        p95s.append(p95*dt + new_time[where][0])

    tas = np.array(tas)
    p05s = np.array(p05s)
    p95s = np.array(p95s)

    keep_indexes = np.argsort(tas)[ndiscard:-ndiscard]
    
    ta_2 = tas[keep_indexes].mean()
    p05_2 = p05s[keep_indexes].mean()
    p95_2 = p95s[keep_indexes].mean()

    return {"arrival": ta_2, "p05": p05_2, "p95": p95_2}