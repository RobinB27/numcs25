import numpy as np
import matplotlib.pyplot as plt
import scipy

"""
    Using Riemann sum directly to calculate the integral (Not a good idea) + Convergence
"""

def piecewise_eval(x: np.ndarray, x_data: np.ndarray, y_data: np.ndarray, f):
    """
        Step-function for Riemann Sum
    """
    condlist = [
        (x_data[i-1] <= x) & (x <= x_data[i])
        for i in range(1, n)
    ]
    funclist =  [
        lambda arg, ind=i: y_data[ind] if abs(y_data[ind]) < abs(y_data[ind-1]) else y_data[ind-1]
        for i in range(1, n)
    ]
    return np.piecewise(x, condlist, funclist)

# Parameters to set

f = lambda x: np.sin(x)
a, b = 0, np.pi
n = 20
n_vals = [10, 20, 30, 40, 50, 100, 250, 500, 1000, 2000, 5000]
N = 1000

# Experiment Runner, specific n with visualisation

x, h = np.linspace(a, b, n, retstep=True)   
y = f(x)

Q_precise = scipy.integrate.quad(f, a, b)[0]
Q_riemann = np.sum( y * h )
err = np.abs( Q_precise - Q_riemann )

print("Quadrature (scipy):\t", Q_precise)
print("Quadrature (Riemann):\t", Q_riemann)
print("Error:\t", err)
print("n:\t", n)
print("h:\t", h)

vals = np.linspace(a, b, N)
fx = f(vals)
fx_approx = piecewise_eval(vals, x, y, f)

# Experiment Runner, convergence analysis for multiple n

err_vals = np.zeros(len(n_vals))
for i, n_val in enumerate(n_vals):
    # Ideally expand this to use multiple, different functions
    x_n, h_n = np.linspace(a, b, n_val, retstep=True)
    y_n = f(x_n)
    Q_riemann_n = np.sum( y_n * h_n )
    err_vals[i] = np.abs( Q_precise - Q_riemann_n )

# Plotting

fig, axs = plt.subplots(1, 3)

axs[0].plot(vals, fx, label="f(x)")
axs[0].scatter(x, y, label="data points")
axs[0].plot(vals, fx_approx, label="Riemann sum")
axs[0].fill_between(vals, fx_approx)

axs[0].set_title("Riemann Integral")
axs[0].grid()
axs[0].legend()

axs[1].semilogy(n_vals, err_vals, label="Error")

axs[1].set_title("Convergence (loglin)")
axs[1].grid()
axs[1].set_xlabel("n")
axs[1].set_ylabel("Error")

axs[2].loglog(n_vals, err_vals, label="Error")

axs[2].set_title("Convergence (loglog)")
axs[2].grid()
axs[2].set_xlabel("n")
axs[2].set_ylabel("Error")

fig.tight_layout()

plt.show()