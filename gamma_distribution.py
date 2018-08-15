from scipy.special import gamma as gamma_fun
import numpy as np

def gamma_pdf(x, a, b):
    return b**a*x**(a-1)*np.exp(-b*x)/gamma_fun(a)

def estimate_a(samples):
    return len(samples)/2.0

def estimate_b(samples):
    return np.sum(samples**2.0)/2.0
