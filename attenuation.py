from scipy.fftpack import rfft, rfftfreq
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from picoscopereader import load_psdata_bufferized

def frequency_domain(signal, dt):
    signal_f = rfft(signal)
    freq = rfftfreq(len(signal), dt)
    return freq, signal_f

def to_dbi(data):
    log10data = np.log10(data)
    maxlog10data = np.max(log10data)
    return 10.0*(log10data - maxlog10data)

def q_estimation(reference_amplitude, sample_amplitude, frequency):
    a = reference_amplitude/sample_amplitude
    slope, intercept = linregress(frequency,a)
    plt.semilogy(frequency, np.abs(a), )
    plt.semilogy(frequency, intercept + slope*frequency, 'r')
    plt.ylabel('LN(A_ref / A_a')
    plt.xlabel('Frequência (kHz)')

def fit_tanh(x, y):
    def target(args):
        a, b, c, d = args
        tanh = a*np.tanh(c*x + d) + b
        # return np.mean((tanh - y)**2.0)
        return np.mean(np.abs(tanh - y))
    
    x0 = [1, np.mean(y), 1, -np.mean(x)]

    result = minimize(target, x0)

    return result.x

if __name__ == '__main__':
    import sys
    import json

    with open(sys.argv[1]) as f:
        info = json.load(f)

        f2f = info["face_to_face_first_break_time"]
        fmin = info["minimum_frequency"]
        fmax = info["maximum_frequency"]
        
        data = {}

        for key in ["reference", "sample"]:
            d = {}
            time, waves = load_psdata_bufferized(info[key]["file"], False, "buffer")
            time *= info[key]["time_conversion_factor"]
            waves *= info[key]["amplitude_conversion_factor"]
            l = info[key]["length"]
            t0 = info[key]["first_break_time"]
            v = l/(t0 - f2f)

            d["time"] = time
            d["data"] = waves[0]
            d["l"] = l
            d["t0"] = t0
            d["v"] = v

            data[key] = d
        
    min_length = min(data[key]["l"] for key in ["reference", "sample"])

    for i, key in enumerate(["reference", "sample"]):
        t0 = data[key]["t0"]
        v = data[key]["v"]
        t1 = t0 + min_length/v
        data[key]["t1"] = t1

        time = data[key]["time"]
        wave = data[key]["data"]

        where = (time >= t0) & (time <= t1)

        wave_f = np.fft.rfft(wave[where])
        freq = np.fft.rfftfreq(np.sum(where), time[1] - time[0])

        data[key]["data_f"] = wave_f
        data[key]["freq"] = freq

        ylim = np.max(np.abs(wave))*1.1
        ylim_f = np.max(np.abs(wave_f))*1.1

        plt.subplot(5, 2, 1 + i)
        plt.plot(time, wave)
        
        plt.vlines([t0, t1], -ylim, ylim, colors='r', linestyles='dashed')
        plt.xlim(0.0, t1 + (t1 - t0)*0.1)
        plt.ylim(-ylim, ylim)
        plt.subplot(5, 2, 3 + i)
        plt.plot(time, wave)
        plt.xlim(t0, t1)
        plt.ylim(-ylim, ylim)

        plt.subplot(5, 2, 5 + i)
        plt.plot(freq, np.abs(wave_f))
        plt.vlines([fmin, fmax], 0, ylim_f, colors='r', linestyles='dashed')
        plt.xlim(0.0, np.max(freq))
        plt.ylim(0, ylim_f)

        plt.subplot(5, 2, 7 + i)
        plt.plot(freq, np.abs(wave_f))
        plt.xlim(fmin, fmax)
        plt.ylim(0, ylim_f)

    ref_data_f_resamp = np.interp(data["sample"]["freq"], data["reference"]["freq"], np.abs(data["reference"]["data_f"]))

    where_f = (data["sample"]["freq"] >= fmin) & (data["sample"]["freq"] <= fmax)

    x = data["sample"]["freq"][where_f]
    y = np.log(np.abs(ref_data_f_resamp)/np.abs(data["sample"]["data_f"]))[where_f]
    a, b, c, d = fit_tanh(x, y)
    y_ = a*np.tanh(c*x + d) + b

    y0 = -a + b
    y1 = a + b
    xc = -d/c
    x0 = (x[0] + xc)/2.0
    x1 = (x[-1] + xc)/2.0

    dydx = (y1 - y0)/(x1 - x0)

    Q = np.pi*min_length/dydx/data["sample"]["v"]

    plt.subplot(5, 1, 5)
    plt.plot(x, y)
    plt.plot(x, y_)
    plt.plot([x0, x1], [y0, y1], label="Q = {:.2f}".format(Q))
    plt.legend()
    plt.show()
