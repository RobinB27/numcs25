import numpy as np
import matplotlib.pyplot as plt

"""
    Newton method for an R^2 -> R^2 function
"""

def newton_2d(x: np.ndarray, F, DF, tol=1e-12, maxIter=50, results=None):
    """ Newton method in 2d using Jacobi Matrix of F"""
    if results is not None: results.append(x.copy())
    for i in range(maxIter):
        s = np.linalg.solve(DF(x[0], x[1]), F(x[0], x[1]))
        x -= s      # s = DF(x^k)^{-1} @ F(x^k)
        if results is not None: results.append(x.copy())
        if np.linalg.norm(s) < tol * np.linalg.norm(x): return x, i
    return x, maxIter

# Parameters to set

F = lambda x, y: np.array([    # Function to use
    x**2 - y**4,
    x - y**3
])
DF = lambda x, y: np.array([   # Jacobi Matrix of F, sympy is useful for calculating these
    [ 2*x, -4*y**3 ],
    [ 1, -3*y**2 ]
])
x  = np.array([0.7, 0.7])   # Starting x, needs to be very close!
x0 = np.array([1, 1])       # Exact solution

maxIter = 50
tolerance = 1e-12

N = 25                     # grid size for plot
a, b = 0.5, 1.5
c, d = 0.5, 1.5

# Experiment Runner

results = []
sol, it = newton_2d(x, F, DF, tolerance, maxIter, results)
err = np.abs( np.linalg.norm(x0) - np.linalg.norm(results, axis=1) )

x, y = np.linspace(a, b, N), np.linspace(c, d, N)
X, Y = np.meshgrid(x, y)
U, V = F(X, Y)

results = np.array(results).T

# Plotting

fig, axs = plt.subplots(1, 3)

axs[0].quiver(X, Y, U, V)
axs[0].plot(results[0], results[1], label="Newton 2D")
axs[0].scatter(results[0][0], results[1][0], label="Start")
axs[0].scatter(results[0][-1], results[1][-1], label=f"End, k={len(results[0])}")
axs[0].scatter(x0[0], x0[1], label="Exact Solution")

axs[0].set_title("Newton Method")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].legend()

axs[1].semilogy(np.arange(1, len(err)+1), err, label="Newton 2D")

axs[1].set_title("Convergence")
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("Error (Normed)")
axs[1].grid()
axs[1].legend()

plt.show()