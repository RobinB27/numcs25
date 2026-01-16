import numpy as np
import matplotlib.pyplot as plt

"""
    Curve fitting via solving normal equations directly (bad idea)
    Since the condition of A^T @ A is bad, normal equations quickly generate large errors
"""

# Parameters to set

A = np.array([      # Good condition
    [98.269,  1],
    [0,       1],
    [-194.96, 1]
])
b = np.array([852.7, 624.5, 172.7])

eps = 1e-7          # Bad condition
B = np.array([
    [1+eps, 1],
    [1-eps, 1],
    [eps, eps]
])
c = np.array([50, 20, 10])

# Experiment runner

print("Condition of A:\t", np.linalg.cond(A))
print("Condition of B:\t", np.linalg.cond(B))

x_norm = np.linalg.solve(A.T @ A, A.T @ b)  # Can also be done with LU, or even Cholesky 
x_lstsq = np.linalg.lstsq(A, b)[0]          # Uses SVD internally

y_norm = np.linalg.solve(B.T @ B, B.T @ c)
y_lstsq = np.linalg.lstsq(B, c)[0]

# Plotting

N = np.linspace(-200, 200, 1000)
fig, axs = plt.subplots(1, 2)

axs[0].plot(N, x_lstsq[0]*N + x_lstsq[1], label="Least Squares fit")
axs[0].plot(N, x_norm[0]*N + x_norm[1], label="Normal Equations fit")
axs[0].scatter(A.T[0], b, label="Data points")

axs[0].set_title("Good Condition")
axs[0].legend()
axs[0].grid()

axs[1].plot(N, y_lstsq[0]*N + y_lstsq[1], label="Least Squares fit")
axs[1].plot(N, y_norm[0]*N + y_norm[1], label="Normal Equations fit")
axs[1].scatter(B.T[0], c, label="Data points")

axs[1].set_title("Bad Condition")
axs[1].legend()
axs[1].grid()

plt.show()