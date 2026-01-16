import numpy as np
import matplotlib.pyplot as plt
import scipy

"""
    A good way to solve linear curve fitting problems is ortho. transformations
    So either QR or SVD 
"""

def least_squares_SVD(A: np.ndarray, b: np.ndarray, eps=1e-6):
    U, S, Vh = np.linalg.svd(A)
    r = 1 + np.where(S / S[0] > eps)[0].max()                   # Approximates rank
    y = np.dot( Vh[:r, :].T, np.dot(U[:, :r].T, b ) / S[:r] )   # Solve
    return y


# Parameters to set

A = np.array([      # Good condition
    [98.269,  1],
    [0,       1],
    [-194.96, 1]
])
b = np.array([852.7, 624.5, 172.7])

eps = 1e-6        # Bad condition
B = np.array([
    [1+eps, 1],
    [1-eps, 1],
    [eps, eps]
])
c = np.array([50, 20, 10])

# Experiment runner

print("Condition of A:\t", np.linalg.cond(A))
print("Condition of B:\t", np.linalg.cond(B))


Q1, R1 = np.linalg.qr(A)
x_qr = np.linalg.solve(R1, np.dot(Q1.T, b) )
x_svd = least_squares_SVD(A, b)       

Q2, R2 = np.linalg.qr(B)
y_qr = np.linalg.solve(R2, np.dot(Q2.T, c) )
y_svd = least_squares_SVD(B, c)

# Plotting

N = np.linspace(-200, 200, 1000)
fig, axs = plt.subplots(1, 2)

axs[0].plot(N, x_svd[0]*N + x_svd[1], label="QR")
axs[0].plot(N, x_qr[0]*N + x_qr[1], label="SVD")
axs[0].scatter(A.T[0], b, label="Data points")

axs[0].set_title("Good Condition")
axs[0].legend()
axs[0].grid()

axs[1].plot(N, y_svd[0]*N + y_svd[1], label="QR")
axs[1].plot(N, y_qr[0]*N + y_qr[1], label="SVD")
axs[1].scatter(B.T[0], c, label="Data points")

axs[1].set_title("Bad Condition")
axs[1].legend()
axs[1].grid()

plt.show()