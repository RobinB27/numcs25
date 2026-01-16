import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_solve, lu_factor, norm, solve

"""
    Broyden Quasi-Newton uses an iterative step 
    to approximate the next Jacobian required.
    This converges slower than proper Newton, but faster than
    re-using the Jacobian (simplified Newton)
"""

def fast_broyden(x0: np.ndarray, F, J, tol=1e-12, maxIter=20, results=None):
    x   = x0.copy()
    lup = lu_factor(J)
    
    if results is not None: results.append(x.copy())
    
    s  = lu_solve(lup, F(x))
    sn = np.dot(s, s)
    x -= s
    
    if results is not None: results.append(x.copy())
    
    # Book keeping, for Broyden Update
    dx     = np.zeros((maxIter, len(x)))
    dxn    = np.zeros(maxIter)
    dx[0]  = s
    dxn[0] = sn
    k = 1
    
    while sn > tol and k < maxIter:
        w = lu_solve(lup, F(x))     # Simplified Newton Update
        
        # Apply Broyden correction (Shermann-Morrison-Woodbury formula)
        for r in range(1, k):
            w += dx[r] * np.dot(dx[r-1], w) / dxn[r-1]
        z = np.dot(s, w)
        s = (1 + z/(sn-z)) * w
        x -= s      # Apply the iteration
        
        if results is not None: results.append(x.copy())
        
        # Book keeping again
        sn = np.dot(s, s)
        dx[k] = s
        dxn[k] = sn
        k += 1
    
    return x, k

# To compare against broyden:
def newton_2d(x: np.ndarray, F, DF, tol=1e-12, maxIter=50, results=None):
    """ Newton method in 2d using Jacobi Matrix of F"""
    if results is not None: results.append(x.copy())
    for i in range(maxIter):
        s = np.linalg.solve(DF(x), F(x))
        x -= s      # s = DF(x^k)^{-1} @ F(x^k)
        if results is not None: results.append(x.copy())
        if np.linalg.norm(s) < tol * np.linalg.norm(x): return x, i
    return x, maxIter

# Parameters to set

x_start = np.array([0.7, 0.7])  # Same problem as in 2d-newton
x_exact = np.array([1, 1])

F  = lambda x: np.array([
    x[0]**2 - x[1]**4,
    x[0]    - x[1]**3
])
DF = lambda x: np.array([
    [2*x[0], -4*x[1]**3],
    [1,      -3*x[1]**2]
])

tolerance = 1e-12
maxIter   = 20

N = 25                     # grid size for plot
a, b = 0.5, 1.5
c, d = 0.5, 1.5

# Experiment Runner

results_broyd = []
sol_broyd, it_broyd = fast_broyden(x_start, F, DF(x_start), tolerance, maxIter, results_broyd)

results_newt = []
sol_newt, it_newt = newton_2d(x_start, F, DF, tolerance, maxIter, results_newt)

print("Exact solution:\t\t", x_exact)
print("Broyden solutiion:\t", sol_broyd)
print("Newton solution:\t", sol_newt)
print("Iterations:\t\t", it_broyd, ", ", it_broyd)

n = len(results_broyd)
errs_broyd = np.zeros(n)
for i in range(n):
    errs_broyd[i] = np.abs( np.linalg.norm(x_exact) - np.linalg.norm(results_broyd[i]) )

errs_newt = np.abs( np.linalg.norm(x_exact) - np.linalg.norm(results_newt, axis=1) )

F_lambd = lambda x, y: F([x, y])    # must do this to use the meshgrid below

x, y = np.linspace(a, b, N), np.linspace(c, d, N)
X, Y = np.meshgrid(x, y)
U, V = F_lambd(X, Y)

# Reshape so they can be plotted

results_broyd = np.array(results_broyd).T
tmp = np.zeros((2, len(results_newt)))
for i in range(len(results_newt)):
    tmp[0][i] = results_newt[i][0]; tmp[1][i] = results_newt[i][1]
results_newt = tmp

# Plotting

fig, axs = plt.subplots(1, 2)

axs[0].quiver(X, Y, U, V)
axs[0].plot(results_newt[0], results_newt[1], label=f"Newton (it={it_newt})")
axs[0].plot(results_broyd[0], results_broyd[1], label=f"Broyden (it={it_broyd})")
axs[0].scatter(results_broyd[0][0], results_broyd[1][0], label="Start")
axs[0].scatter(results_broyd[0][-1], results_broyd[1][-1], label="End")
axs[0].scatter(x_exact[0], x_exact[1], label="Exact Solution")

axs[0].set_title("Broyden Method")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].legend()

axs[1].semilogy(np.arange(1, len(errs_broyd)+1), errs_broyd, label="Broyden")
axs[1].semilogy(np.arange(1, len(errs_newt)+1), errs_newt, label="Newton")

axs[1].set_title("Convergence")
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("Error")
axs[1].grid()
axs[1].legend()

fig.tight_layout()

plt.show()
