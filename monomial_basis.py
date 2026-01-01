import numpy as np
import matplotlib.pyplot as plt

""" 
    Setup
    Choose function f, and points x to use for polynomial
"""

points_to_try = 6
f = lambda x: 1/(1 + 5*x**2)    # Runge function

# Compute approx. poly. by solving Vander. matrix (like np.polyfit)

x = np.linspace(-1, 1, points_to_try, endpoint=True);
y = f(x)

A = np.vander(x)
alpha = np.linalg.solve(A, y)

# functionally equivalent to np.polyval
def horner(coeffs: np.ndarray, vals: np.ndarray):
    ret = coeffs[0]
    for i in range(1, len(coeffs)): ret = vals * ret + coeffs[i]
    return ret

n = np.linspace(-1, 1, 1000)
fig, ax = plt.subplots()
ax.plot(n, f(n), label="f")
ax.plot(n, horner(alpha, n), label="p")
ax.legend()
ax.set_title("Monomial basis approx.")
plt.show()