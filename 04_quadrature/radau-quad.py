import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_jacobi
from scipy.integrate import quad

"""
    Radau Quadrature, variation of Gauss which includes one specific point too.
"""

def radau_quad(f, s:int):
    """
        Radau Quadrature: Fixes one specific point c (here x=1), otherwise Gauss weights/nodes
        Uses reference interval [-1, 1]
    """
    c, w = roots_jacobi(s-1, alpha=1, beta=0)
    b = np.zeros(s)
    b[0:s-1] = w / (1-c)
    b[s-1] = 2 / s**2 
    Q_radau = np.sum( b[0:s-1]*f(c) ) + b[s-1]*f(1)
    return Q_radau

# Parameters to set

f = lambda x: 1 / (1 + (5*x)**2)
s = 20
s_vals = np.arange(2, 201)
N = 1000

# Experiment Runner

Q_precise = quad(f, -1, 1)[0]
Q_radau = radau_quad(f, s)
err = np.abs( Q_precise - Q_radau )

print("Error:\t", err)
print("s:\t", s)

x = np.linspace(-1, 1, N)
y = f(x)

err_vals = np.zeros(s_vals.size)
for i, s_val in enumerate(s_vals):
    Q_radau_s = radau_quad(f, s_val)
    err_vals[i] = np.abs( Q_precise - Q_radau_s )

# Plotting

fig, axs = plt.subplots(1, 2)

axs[0].plot(x, y)

axs[0].set_title("Function")
axs[0].grid()

axs[1].semilogy(s_vals, err_vals, label="Radau")

axs[1].set_title("Convergence")
axs[1].grid()
axs[1].set_ylabel("Error")
axs[1].set_xlabel("s")
axs[1].legend()

fig.tight_layout()

plt.show()