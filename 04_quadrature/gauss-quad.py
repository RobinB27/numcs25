import numpy as np
import matplotlib.pyplot as plt
import scipy

"""
    Non-equidistant Gauss Quadrature using Golub & Welsh algorithm
    Simpson quadrature for comparison, too.
"""

def gauss_quad(n: int):
    """ Computes nodes & weights for Gauss Quadrature (Golub & Welsh)"""
    i = np.arange(1, n)
    b = i / np.sqrt(4*i**2 - 1)
    J = np.diag(b, -1) + np.diag(b, 1)  # Symmetric Matrix
    x, ev = np.linalg.eigh(J)           # Eigenvalues/vectors of Matrix
    w = 2 * ev[0,:]**2                  # Formula for gauss weights
    return x, w


def simpson(f, a: float, b: float, N: int):
    """ Quadrature via simpson rule """
    x, h = np.linspace(a, b, 2*N+1, retstep=True)
    Q = h/3 * np.sum( f(x[:-2:2]) + 4*f(x[1:-1:2]) + f(x[2::2]) )
    return Q 

# Parameters to set

f = lambda x: 1 / (1 + (x+5)**2)
a, b = -1, 1    # Must be this interval for Gauss. To integrate other interval, modify function instead
n = 100
N = 1000

# Experiment Runner

n_vals = np.arange(1, 101)
err_gauss = np.zeros(n_vals.size)
err_simp  = np.zeros(n_vals.size)

Q_precise = scipy.integrate.quad(f, a, b)[0]
for i, n_val in enumerate(n_vals):
    x, w = gauss_quad(n_val)
    Q_gauss = np.sum( f(x)*w )
    Q_simp  = simpson(f, a, b, n_val)
    err_gauss[i] = np.abs( Q_precise - Q_gauss )
    err_simp[i]  = np.abs( Q_precise - Q_simp )

p = np.polyfit(np.log(n_vals), np.log(err_gauss), 1)[0]
p = np.abs( p )

h2 = ( np.abs(a-b)/n_vals )**2
h4 = ( np.abs(a-b)/n_vals )**4

print("Convergence Order (Gauss):\t", p)

x = np.linspace(a, b, N)
y = f(x)

# Plotting

fig, axs = plt.subplots(1, 2)

axs[0].plot(x, y, label="f(x)")

axs[0].set_title("Function")
axs[0].grid()

axs[1].semilogy(n_vals, err_gauss, label="Gauss Quadrature")
axs[1].semilogy(n_vals, err_simp, label="Simpson Quadrature")
axs[1].semilogy(n_vals, h2, label="h^2", linestyle="dashed", color="lightgray")
axs[1].semilogy(n_vals, h4, label="h^4", linestyle="dashed", color="lightgray")

axs[1].set_title("Convergence")
axs[1].set_xlabel("n")
axs[1].set_ylabel("Error")
axs[1].grid()
axs[1].legend()

fig.tight_layout()

plt.show()
