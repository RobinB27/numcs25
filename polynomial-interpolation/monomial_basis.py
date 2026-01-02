import numpy as np
import matplotlib.pyplot as plt

def coeffs_monomial(x: np.ndarray, y: np.ndarray):
    """
        Simply solve Vandermonde matrix. This is very unstable.
    """
    A = np.vander(x)
    alpha = np.linalg.solve(A, y)
    return alpha


def eval_horner(coeffs: np.ndarray, vals: np.ndarray):
    """
        Evaluate polynomial using Horner scheme
    """
    ret = coeffs[0]
    for i in range(1, len(coeffs)): ret = vals * ret + coeffs[i]
    return ret


# Parameters to set

points_to_try = 6
f = lambda x: 1/(1 + 5*x**2)    # Runge function

# Experiment runner

x = np.linspace(-1, 1, points_to_try, endpoint=True);   # equidistant points
y = f(x)
coeffs = coeffs_monomial(x, y)

# Plotting

n = np.linspace(-1, 1, 1000)
fig, ax = plt.subplots()

ax.plot(n, f(n), label="f")
ax.plot(n, eval_horner(coeffs, n), label="p")
ax.scatter(x, y, label="x")

ax.legend()
ax.set_title("Monomial basis approx.")
plt.show()