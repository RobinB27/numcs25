import numpy as np
import matplotlib.pyplot as plt
import scipy

def gram_schmidt(A: np.ndarray):
    """ Regular Gram-Schmidt, assumes lin. indep. columns in A """
    m, n = A.shape
    Q, R = np.zeros((m, n)), np.zeros((m, n))
    
    for j in range(0, n):
        Q[:, j] = A[:, j]
        for i in range(0, j):
            proj = np.dot(A[:, j], Q[:, i])
            R[i, j] = proj
            Q[:, j] = Q[:, j] - proj * Q[:, i]
        Q[:, j] = Q[:, j] / np.linalg.norm(Q[:, j], 2)
    
    return Q, R


def gram_schmidt_mod(A: np.ndarray):
    """ Gram-Schmidt in place, assumes lin. indep. columnes in A  """
    m, n = A.shape
    Q, R = np.zeros((m, n)), np.zeros((m, n))
    
    for i in range(1, n):
        norm = np.linalg.norm(A[:, i])
        Q[:, i] = A[:, i] / norm
        
        for j in range(i+1, n):
            proj = np.dot(A[:, j], Q[:, i])
            A[:, j] = A[:, j] - proj * Q[:, i]
            R[i, j] = proj
        
        R[i, i] = np.dot( Q[:, i], A[:, i] )
    
    return Q, R

# Parameters to set

m, n = 50, 50

A = np.zeros((m, n))    # Matrix to use
for i in range(m):
    for j in range(n):
        A[i, j] = 1 + min(i, j)

# Experiment Runner

print("Matrix Condition:\t", np.linalg.cond(A))

q1, r1 = np.linalg.qr(A)
q2, r2 = gram_schmidt(A)
q3, r3 = gram_schmidt_mod(A)

# Plotting

fig, axs = plt.subplots(1, 3)

fig.suptitle("Gram Schmidt Variations, Error")

axs[0].set_title("Numpy QR")
axs[0].imshow( 
    np.log10( 
        np.abs( 
            np.dot(q1.T, q1) - np.eye(n) 
        )
    + 1e-16),
    vmin= -16, vmax = 1, interpolation="nearest"
)

axs[1].set_title("Gram Schmidt")
axs[1].imshow( 
    np.log10( 
        np.abs( 
            np.dot(q2.T, q2) - np.eye(n) 
        )
    + 1e-16),
    vmin= -16, vmax = 1, interpolation="nearest"
)

axs[2].set_title("Gram Schmidt mod.")
axs[2].imshow( 
    np.log10( 
        np.abs( 
            np.dot(q3.T, q3) - np.eye(n) 
        )
    + 1e-16),
    vmin= -16, vmax = 1, interpolation="nearest"
)

fig.tight_layout()

plt.show()