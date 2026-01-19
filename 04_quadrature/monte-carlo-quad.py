import numpy as np
import matplotlib.pyplot as plt
import scipy
import math

"""
    Monte Carlo integration, example function is the n-dimensional unit sphere
"""

def montecarlo_quad(f, a: float, b: float, N: int, rng):
    """ n-dimensional quadrature via monte carlo method (guessing) """
    if ( len(a) != len(b) ): return
    dim = len(a)
    
    x = rng.random((dim, N))
    fx = f(x)
    Q = 1/N * np.sum(fx)    # Expected Value of Integral
    
    vol = 1
    for i in np.arange(dim): vol *= (b[i] - a[i])
    return vol * Q


def montecarlo_experiment(f, a: float, b: float, N: int, M: int, rng):
    """ Performs M mc experiments using given params """
    Q = np.zeros(M)
    for i in np.arange(M):
        Q[i] = montecarlo_quad(f, a, b, N, rng)
        
    Q_mean = np.sum(Q) / M
    variance = np.sum(Q**2) / (M-1) - (M/(M-1))*(Q_mean**2)
    
    return Q_mean, variance
    

def f(x):
    """ Unit sphere in d dimensions """
    dim, n = np.shape(x)
    r2 = 0
    for i in np.arange(dim): r2 += x[i, :]**2
    y = r2 <= 1    # f(x) = 1 <=> x inside unit sphere
    return y


# Parameters to set

dim = 4
a = -np.ones(dim)
b = +np.ones(dim)
M = 100                          # Experiment count (M > 1), quickly takes long to run
N_vals = 4*10**np.arange(1, 7)   # Sample counts

# Experiment Runner

Q_exact = np.pi**(dim/2) / math.gamma(dim/2 + 1)    # Analytic vol of unit sphere
rng = np.random.default_rng()

Q_mean = []; Q_var = []; err = []
for N in N_vals:
    Q, var = montecarlo_experiment(f, a, b, N, M, rng)
    Q_mean.append(Q)
    Q_var.append(var)
    err.append(np.abs( Q - Q_exact ))

# Plotting

fig, ax = plt.subplots()

ax.loglog(N_vals, err, label=f"avg over m={str(M)} runs")
ax.loglog(N_vals, 1/np.sqrt(N_vals), label="O(sqrt(N))", linestyle="dashed")

ax.set_title(f"Error in {dim}d Monte Carlo Quad.")
ax.set_xlabel("Sample count")
ax.set_ylabel("Error")
ax.legend()
ax.grid()

fig.tight_layout()

plt.show()