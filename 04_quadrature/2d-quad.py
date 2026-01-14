import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy

"""
    Quadrature on function taking 2 arguments
"""

def trapezoid_1d(f, a: float, b: float, y0: float, N: int):
    """ 1D Quadrature via trapezoidal rule, accepting fixed y0 """
    x, h = np.linspace(a, b, N+1, retstep=True)
    Q = f(x[0], y0) + f(x[-1], y0)
    Q += 2 * np.sum( f(x[1:-1], y0) )
    Q *= h/2
    return Q


def trapezoid_2d(f, a: float, b: float, Nx: int, c: float, d: float, Ny: int):
    """ 2D Quadrature via trapezoidal rule, by reusing the 1d function """
    y, h = np.linspace(c, d, Ny+1, retstep=True)
    Q =  1.0 * trapezoid_1d(f, a, b, y[0], Nx)
    Q += 1.0 * trapezoid_1d(f, a, b, y[-1], Nx)
    for i in range(1, Ny):
        Q += 2.0 * trapezoid_1d(f, a, b, y[i], Nx)
    return h/2 * Q


def trapezoid_2d_mesh(f, a: float, b: float, Nx: int, c: float, d: float, Ny: int):
    """ Trapezoidal rule on a 2d function via np.meshgrid """
    x, hx = np.linspace(a, b, Nx+1, retstep=True)
    y, hy = np.linspace(c, d, Ny+1, retstep=True)
    
    X, Y = np.meshgrid(x, y)
    F = f(X, Y)
    
    # Apply once along x-axis, once along y-axis
    Q_x = hx/2 * ( 2.0 * np.sum(F[:, 1:-1], axis=1) + F[:, 0] + F[:, -1] )
    Q_y = hy/2 * ( 2.0 * np.sum( Q_x[1:-1] ) + Q_x[0] + Q_x[-1] )
    
    return Q_y


def simpson_2d_weights(N: int):
    """ Generate weights for simpson rule """
    w = np.zeros(N+1)
    w[0] = 1.0; w[-1] = 1.0
    for i in range(1, N):
        w[i] = 2.0 if i % 2 == 0 else 4.0
    return w


def simpson_2d_mesh(f, a: float, b: float, Nx: int, c: float, d: float, Ny: int):
    """ Simpson rule on a 2d function via np.meshrgdi """
    x, hx = np.linspace(a, b, Nx+1, retstep=True)
    y, hy = np.linspace(c, d, Ny+1, retstep=True)
    
    X, Y = np.meshgrid(x, y)
    F = f(X, Y)
    wx, wy = simpson_2d_weights(Nx), simpson_2d_weights(Ny)
    W = np.outer(wx, wy)
    
    scale = hx*hy / 3**2
    Q = scale * np.sum( W * F )
    return Q


# Parameters to set

f = lambda x, y: x**3 + np.sin(x) + 1/np.cos(y)
a, b = -1, 1
c, d = -1, 1
N = 1000            
n_max = 1000     # finest grid size to use, takes long to produce graph

n_vals = np.arange(2, n_max, 2) # simpson needs even values

# Experiment Runner

x = np.linspace(a, b, N)
y = np.linspace(c, d, N)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

Q_precise = scipy.integrate.nquad(f, [[a, b], [c, d]])[0]

err = np.zeros((2, n_vals.size))
for i, n in enumerate(n_vals):
    if (i % 100 == 0): print(f"Finished:\t {i}")
    Q_trap = trapezoid_2d(f, a, b, n, c, d, n)
    Q_simp = simpson_2d_mesh(f, a, b, n, c, d, n)
    err[0, i] = np.abs( Q_precise - Q_trap )
    err[1, i] = np.abs( Q_precise - Q_simp )
print("Finished all quadratures")

# Plotting

fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
surf = ax1.plot_surface(X, Y, Z, cmap=plt.cm.cividis)
fig.colorbar(surf, shrink=0.5, aspect=8)

ax2 = fig.add_subplot(1, 2, 2)
ax2.semilogy(n_vals, err[0, :], label="Trapezoidal")
ax2.semilogy(n_vals, err[1, :], label="Simpson")

ax2.set_title("Convergence")
ax2.set_xlabel("n")
ax2.set_ylabel("Error")
ax2.grid()
ax2.legend()

fig.tight_layout()

plt.show()