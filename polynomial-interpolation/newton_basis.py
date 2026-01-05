import numpy as np
import matplotlib.pyplot as plt

"""
    Divided differences to generate coeffs for the Newton poly.
"""

def divdiff_seq(x: np.ndarray, y: np.ndarray):
    """
        Sequential version, no vectorization
    """
    n = y.size
    z = y.copy()
    for i in range(1, n):
        # Must loop down, since we need the previous z[j-1]
        for j in range(n-1, i-1, -1):   
            z[j] = (z[j] - z[j-1]) / (x[j] - x[j-i])
    return z


def divdiff_mat(x: np.ndarray, y: np.ndarray):
    """
        Passes through a Matrix column-wise to calculate div. diff.
    """
    n = y.size
    T = np.zeros((n,n))
    T[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            T[i, j] = ( T[i+1, j-1] - T[i, j-1] ) / ( x[i+j] - x[i] )
    return T[0,:]


def divdiff_vec(x: np.ndarray, y: np.ndarray):
    """
        Same as matrix version, but on a single vector
    """
    n = y.size
    z = np.copy(y)
    for j in range(1, n):
        z[j:n] = (z[j:n] - z[j-1:n-1]) / (x[j:n] - x[:n-j])
    return z

def coeffs_newton(x: np.ndarray, y: np.ndarray):
    """
        Divided differences gives the required coefficients
    """
    return divdiff_vec(x, y)
        

def test_divdiff(x: np.ndarray, y: np.ndarray):
    """
        Test all diff div version on same input. 
        Output should be identical up to floating point inaccuracies.
    """
    print("seq:")
    print(divdiff_seq(x, y))
    print("mat:")
    print(divdiff_mat(x, y))
    print("vec:")
    print(divdiff_vec(x, y))
    

def eval_horner_newton(coeffs: np.ndarray, data: np.ndarray, x: np.ndarray):
    n = coeffs.size
    p = coeffs[n-1]
    for i in range(n-2, -1, -1):
        p = (x - data[i])*p + coeffs[i]
    return p

# Parameters to set

points_to_try = 5
f = lambda x: 1/(1 + 5*x**2)    # Runge function

# Experiment runner

x = np.linspace(-1, 1, points_to_try, endpoint=True);   # equidistant points
y = f(x)
coeffs = coeffs_newton(x, y)

# Plotting

n = np.linspace(-1, 1, 1000)
fig, ax = plt.subplots()

ax.plot(n, f(n), label="f")
ax.plot(n, eval_horner_newton(coeffs, x, n), label="p")
ax.scatter(x, y, label="x")

ax.legend()
ax.set_title("Newton basis approx.")
plt.show()
