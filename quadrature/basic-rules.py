import numpy as np
import matplotlib.pyplot as plt
import scipy

"""
    Convergence rates of 3 basic quadrature rules
"""

def midpoint(f, a: float, b: float, N: int):
    """ Quadrature via midpoint rule """
    x, h = np.linspace(a, b, N+1, retstep=True)
    Q = h * np.sum(f( (x[0:N] + x[1:N+1]) / 2 ))
    return Q

def trapezoidal(f, a: float, b: float, N: int):
    """ Quadrature via trapezoidal rule  """
    x, h = np.linspace(a, b, N+1, retstep=True)
    Q = f(x[0]) + f(x[-1])
    Q += 2 * np.sum( f(x[1:-1]) )
    Q *= h/2
    return Q


def simpson(f, a: float, b: float, N: int):
    """ Quadrature via simpson rule """
    x, h = np.linspace(a, b, 2*N+1, retstep=True)
    Q = h/3 * np.sum( f(x[:-2:2]) + 4*f(x[1:-1:2]) + f(x[2::2]) )
    return Q 

# Parameters to set

f = lambda x: 1 / (1 + (5*x)**2)    # Runge
a, b = 0, 1
n = 1000    # max amaount of quadrature intervals to try
N = 1000    # points to plot function on

# Experiment Runner

n_vals = np.arange(2, n+1)
err = np.zeros( (3, len(n_vals)) )

Q_precise = scipy.integrate.quad(f, a, b)[0]
for i, n_val in enumerate(n_vals):
    Q_mid = midpoint(f, a, b, n_val)
    Q_trap = trapezoidal(f, a, b, n_val)
    Q_simp = simpson(f, a, b, n_val)
    err[0, i] = np.abs( Q_precise - Q_mid )
    err[1, i] = np.abs( Q_precise - Q_trap )
    err[2, i] = np.abs( Q_precise - Q_simp )

h2 = ( np.abs(a-b)/n_vals )**2
h4 = ( np.abs(a-b)/n_vals )**4

p = np.zeros(3)
for i in range(3):
    p[i] = np.polyfit(np.log(n_vals), np.log(err[i,:]), 1)[0]
    p[i] = np.abs( p[i] )

print("Convergence Order (Midpoint):\t", p[0])
print("Convergence Order (Trapezoid):\t", p[1])
print("Convergence Order (Simpson):\t", p[2])

# Plotting

fig, axs = plt.subplots(1, 2)

x = np.linspace(a, b, N)
y = f(x)

axs[0].plot(x, y)

axs[0].set_title("Function")
axs[0].grid()
axs[0].legend()

axs[1].semilogy(n_vals, err[0,:], label="Midpoint")
axs[1].semilogy(n_vals, err[1,:], label="Trapezoidal")
axs[1].semilogy(n_vals, err[2,:], label="Simpson")

axs[1].semilogy(n_vals, h2, label="h^2", linestyle="dashed", color="lightgray")
axs[1].semilogy(n_vals, h4, label="h^4", linestyle="dashed", color="lightgray")

axs[1].set_title("Convergence")
axs[1].set_xlabel("n")
axs[1].set_ylabel("Error")
axs[1].legend()
axs[1].grid()

fig.tight_layout()

plt.show()