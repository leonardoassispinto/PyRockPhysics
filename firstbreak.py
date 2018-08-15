import arrival_distribution as ar_dis
import gamma_distribution as gamma
import method as met
import picoscopereader as psdata
import prior_parameters as prior
import numpy as np
import sys
import matplotlib.pyplot as plt
import plug_parameter as plug
import attenuation as at
import os

def first_break2(time, wave, t0=0.0, nchains=5, ndiscard=1, length=None, vmin=None, vmax=None, run_twice=False):
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
    for j in range(nchains):
        arrival_samples = met.infer_arrival(wave[where], time[where], arrival_prior_probs[where], factor, chainsize=10000, burnin=2000, thin=4, useboxcox=False)
        arrival = met.summarize(arrival_samples)['map']

        tas.append(arrival*dt + time[where][0])

    ta_1 = sum(sorted(tas)[ndiscard:-ndiscard])/(nchains - 2.0*ndiscard)

    if not run_twice or (length is None or vmin is None or vmax is None):
        return ta_1

    position = np.argmin(np.abs(ta_1 - time))
    window_size = 50
    p_start = max(0, position-window_size)
    p_end = min(len(time), position+window_size)
    new_time = time[p_start:p_end]
    new_wave = wave[p_start:p_end]

    where = (new_time >= 0.0)
    arrival_prior_probs = met.prior_parameters.tau_trap(new_time, length, vmin_hard, vmin, vmax_hard, vmax)

    tas = []

    for k in range(nchains):
        arrival_samples = met.infer_arrival(new_wave[where], new_time[where], arrival_prior_probs[where], 0.01, chainsize=10000, burnin=2000, thin=4, useboxcox=False)
        arrival = met.summarize(arrival_samples)['map']

        tas.append(arrival*dt + new_time[where][0])

    ta_2 = sum(sorted(tas)[ndiscard:-ndiscard])/(nchains - 2.0*ndiscard)

    return ta_2

def first_break(face_to_face_files, sample_files, length_sample, sample_type):
    # Face-to-face must be the first file
    tff = 0.0
    tas_firstrun = {}

    # m/s
    velocities = plug.standard_velocities(sample_type, 'material_velocities.csv')

    Vmin = velocities[0]
    Vmax = velocities[1]
    Vmin_hard = Vmin - 0.2*(Vmax-Vmin)
    Vmax_hard = Vmax + 0.2*(Vmax-Vmin)

    # converting to mm/us
    Vmin_hard /= 1000.0
    Vmin /= 1000.0
    Vmax /= 1000.0
    Vmax_hard /= 1000.0

    nchains = 5

    # discarded chains in each tip
    ndiscard = 1

    for j, face_to_face_file in enumerate(face_to_face_files):
        time, waves = psdata.load_psdata_bufferized(face_to_face_file, False, "buffer")
        t0 = time[0]
        dt = time[1] - time[0]
        where = (time >= 0.0)*(time <= 40.0)

        # It is not possible determine the tau's prior distribution to face-to-face
        arrival_prior_probs = np.zeros_like(time)
        arrival_prior_probs[where] = 1.0
        arrival_prior_probs /= np.sum(arrival_prior_probs)

        nwaves = waves.shape[0]
        for i, wave in enumerate(waves[:1], 1):
            tas = []

            for j in range(nchains):
                arrival_samples = met.infer_arrival(wave[where], time[where], arrival_prior_probs[where], 0.05, chainsize=10000, burnin=2000, thin=4, useboxcox=False)
                arrival = met.summarize(arrival_samples)['map']

                tas.append(arrival*dt + time[where][0])

            ta = sum(sorted(tas)[ndiscard:-ndiscard])/(nchains - 2.0*ndiscard)

            tff = ta
            label = 'Face-to-Face Time = {0:.2f} us'.format(tff)

            plt.subplot(nwaves, 1, i)
            plt.plot(time, wave, "C7")
            plt.vlines(ta, np.nanmin(wave), np.nanmax(wave), colors='C0', label=label)
            plt.xlim(0.0, np.nanmax(time))
            plt.legend()

        tas_firstrun[face_to_face_file] = ta

    for k, sample_file in enumerate(sample_files):
        time, waves = psdata.load_psdata_bufferized(sample_file, False, "buffer")
        time -= tff
        t0 = time[0]
        dt = time[1] - time[0]
        nwaves = waves.shape[0]
        plt.figure()

        for i, wave in enumerate(waves[:1], 1):
            where = (time >= 0.0)
            arrival_prior_probs = met.prior_parameters.tau_trap(time, length_sample, Vmin_hard, Vmin, Vmax_hard, Vmax)

            tas = []

            for j in range(nchains):
                arrival_samples = met.infer_arrival(wave[where], time[where], arrival_prior_probs[where], 0.01, chainsize=10000, burnin=2000, thin=4, useboxcox=False)
                arrival = met.summarize(arrival_samples)['map']

                tas.append(arrival*dt + time[where][0])

            ta_1 = sum(sorted(tas)[ndiscard:-ndiscard])/(nchains - 2.0*ndiscard)

            position = np.argmin(np.abs(ta_1 - time))
            window_size = 50
            p_start = max(0, position-window_size)
            p_end = min(len(time), position+window_size)
            new_time = time[p_start:p_end]
            new_wave = wave[p_start:p_end]

            where = (new_time >= 0.0)
            arrival_prior_probs = met.prior_parameters.tau_trap(new_time, length_sample, Vmin_hard, Vmin, Vmax_hard, Vmax)

            tas = []

            for b in range(nchains):
                arrival_samples = met.infer_arrival(new_wave[where], new_time[where], arrival_prior_probs[where], 0.01, chainsize=10000, burnin=2000, thin=4, useboxcox=False)
                arrival = met.summarize(arrival_samples)['map']

                tas.append(arrival*dt + new_time[where][0])

            ta_2 = sum(sorted(tas)[ndiscard:-ndiscard])/(nchains - 2.0*ndiscard)

            V = length_sample/ta
            V *= 1000.0
            print("\nSample file: {}\nFirst break: {:.2f} us\nVelocity: {:.2f} m/s\n".format(sample_file, ta_2 + tff, V))
            label = 'First Break Time = {:.2f} us\nVelocity = {:.2f} m/s'.format(ta_2 + tff, V)

            plt.subplot(nwaves, 1, i)
            plt.plot(time, wave, "C7")
            plt.vlines(ta_2, np.nanmin(wave), np.nanmax(wave), colors='C0', label=label)
            plt.xlim(0.0, np.nanmax(time))
            plt.legend()

        tas_firstrun[sample_file] = ta_2

    plt.show()