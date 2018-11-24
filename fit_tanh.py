#fit_tanh.py

from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

def fit_tanh(x, y):
    def target(args):
        a, b, c, d = args
        tanh = a*np.tanh(c*x + d) + b
        return np.mean(np.abs(tanh - y))
    
    x0 = [1, np.mean(y), 1, -np.mean(x)]

    result = minimize(target, x0)

    return result.x