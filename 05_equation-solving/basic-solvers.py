import numpy as np
import matplotlib.pyplot as plt
import scipy

"""
    Bisection, Newton & Secant method for finding zeros to 1d function & Convergence plot
"""

def sgn(x): return 1 if x >= 0 else -1

def bisection_method(f, a: float, b: float, tol=1e-12, maxIter=100, results=None):
    """ Bisection method on f using initial bounds a,b """
    if a > b: a, b = b, a
    if sgn(f(a)) == sgn(f(b)): raise ValueError("f(a) & f(b) must have different signs for bisection")
    
    x = 0.5 * (a + b)
    if results is not None: results += [x]
    
    iter = 1
    while (b-a > tol and a<x and x<b and iter<maxIter):
        # Check on which side f(x) is, update bounds
        if sgn(f(b))*f(x) > 0: b = x
        else:               a = x
        
        x = 0.5 * (a + b)
        if results is not None: results += [x]
        
        iter += 1
        
    return x, iter


def newton_method(f, df, x: float, tol=1e-12, maxIter=100, results=None):
    """ Use Newton's method to find zeros, requires derivative of f """
    iter = 0
    while (np.abs(df(x)) > tol and iter < maxIter):
        x -= f(x)/df(x)
        if results is not None: results += [x]
        iter += 1

    return x, iter


def secant_method(f, x0: float, x1: float, tol=1e-12, maxIter=100, results=None):
    """ Use secant method, which approximates the derivative """
    f0 = f(x0)
    for i in range(maxIter):
        fn = f(x1)
        secant = fn * (x1-x0) / (fn - f0)   # Approximate derivative via secant
        
        x0 = x1; x1 -= secant
        if results is not None: results += [x1]
        
        if np.abs(secant) < tol: return x1, i
        else: f0 = fn
        
    return None, maxIter


# Parameters to set

x0   = 0.1                                      # Initial guess
f    = lambda x: np.exp(2*x) - np.sin(x) -2     # Function
df   = lambda x: 2 * np.exp(2*x) - np.cos(x)    # Derivative, for Newton method
a, b = 0, 1                                     # Bounds for bisection, secant method
N    = 100                                      # Max iterations
tol  = 1e-12                                    # Tolerance

# Experiment Runner

x_bis = []; x_newt = []; x_sec = []

root      = scipy.optimize.root(f, x0)
root_bis  = bisection_method(f, a, b, tol, N, x_bis)
root_newt = newton_method(f, df, x0, tol, N, x_newt)
root_sec  = secant_method(f, a, b, tol, N, x_sec)

print("Scipy Converged") if root.success else print("Scipy didn't converge")
print("Scipy:\t", root.x[0], "\nBisect:\t", root_bis[0], "\nNewton:\t", root_newt[0], "\nSecant:\t", root_sec[0])

# Plotting

fig, ax = plt.subplots()

ax.semilogy(np.arange(1, len(x_bis)+1),  np.abs( x_bis - root.x[0]),  label="Bisection")
ax.semilogy(np.arange(1, len(x_newt)+1), np.abs( x_newt - root.x[0]), label="Newton")
ax.semilogy(np.arange(1, len(x_sec)+1),  np.abs( x_sec - root.x[0]),  label="Secant")

title = "Convergence" if root.success else "Warning: Scipy didnt' converge"
ax.set_title(title)
ax.grid()
ax.legend()

fig.tight_layout()

plt.show()