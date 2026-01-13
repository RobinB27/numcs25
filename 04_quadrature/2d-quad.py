import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy

"""
    Quadrature on function taking 2 arguments
"""


def trapezoidal(f, a: float, b: float, y0: float, N: int):
    """ 1D Quadrature via trapezoidal rule, accepting fixed y0 """
    x, h = np.linspace(a, b, N+1, retstep=True)
    Q = f(x[0], y0) + f(x[-1], y0)
    Q += 2 * np.sum( f(x[1:-1], y0) )
    Q *= h/2
    return Q


def trapezoid_2d(f, a: float, b: float, Nx: int, c: float, d: float, Ny: int):
    """ 2D Quadrature via trapezoidal rule """
    y, h = np.linspace(c, d, Ny+1, retstep=True)
    Q = 0
    for i in range(Ny+1):
        Q += trapezoidal(f, a, b, y[i], Nx)
    return h * Q


# Parameters to set

f = lambda x, y: np.sin(x) + y**2
a, b = -1, 1
c, d = -1, 1
N = 1000

n_vals = np.arange(2, 200)

# Experiment Runner

x = np.linspace(a, b, N)
y = np.linspace(c, d, N)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

Q_precise = scipy.integrate.nquad(f, [[a, b], [c, d]])[0]

err = np.zeros(n_vals.size)
for i, n in enumerate(n_vals):
    Q_trap = trapezoid_2d(f, a, b, n, c, d, n)
    err[i] = np.abs( Q_precise - Q_trap )

# Plotting

fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
surf = ax1.plot_surface(X, Y, Z, cmap=plt.cm.cividis)
fig.colorbar(surf, shrink=0.5, aspect=8)

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(n_vals, err, label="Trapezoidal")

ax2.set_title("Convergence")
ax2.set_xlabel("n")
ax2.set_ylabel("Error")
ax2.grid()
ax2.legend()

fig.tight_layout()

plt.show()