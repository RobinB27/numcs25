import numpy as np
import matplotlib.pyplot as plt

"""
    The SVD can be used to extrapolate information from measurements (with error).
    This is called Principal component analysis.
    
    Here we take 2 signals and add an error, using the SVD we can recover the original signal.
"""

# Parameters to set

n = 10; m = 50
r = np.linspace(1, m, m)

x = np.sin( np.pi * r/m )
y = np.cos( np.pi * r/m )

A = np.zeros((2*n, m))
for k in range(n):
    A[2*k,   :] = x * np.random.rand(m)
    A[2*k+1, :] = y + 0.1*np.random.rand(m)

# Experiment Runner

U, S, Vh = np.linalg.svd(A)
I = np.argsort(1/S)         # Indices, which order singular values descending
Vh1_contrib = np.abs( U[:, 0] )
Vh2_contrib = np.abs( U[:, 1] )

# Plotting: some measurements

fig, axs = plt.subplots(2, 2)

axs[0][0].plot(r, A[0, :], "x", label="Measure 1")
axs[0][0].plot(r, A[1, :], "o", label="Measure 2")
axs[0][0].plot(r, A[2, :], "*", label="Measure 3")
axs[0][0].plot(r, A[3, :], "v", label="Measure 4")

axs[0][0].set_title(f"4/{2*n} Measurements")
axs[0][0].legend()

# Plotting: Weight of principal components (via singular values)

axs[0][1].bar(np.arange(1, len(S)+1), S[I])

axs[0][1].set_title("Singular Values")
axs[0][1].set_xlabel("Singular value number")
axs[0][1].set_ylabel("Singular value")

# Plotting: Largest principal components (Rows of Vh)

axs[1][0].plot(r, -Vh[0, :], "o", label="1st principal comp.")
axs[1][0].plot(r, y / np.linalg.norm(y), "+", label="1st model vector")
axs[1][0].plot(r, Vh[1, :], "o", label="2nd principal comp.")
axs[1][0].plot(r, x / np.linalg.norm(x), "+", label="2nd model vector")

axs[1][0].set_title(f"Largest principal components")
axs[1][0].legend()

# Plotting: Contribution of principal components (Cols of U)

axs[1][1].plot(np.arange(1,2*n+1), Vh1_contrib, "+", label="1st principal comp.")
axs[1][1].plot(np.arange(1,2*n+1), Vh2_contrib, "o", label="2nd principal comp.")

axs[1][1].set_xlabel("Measurement")
axs[1][1].set_ylabel("Contribution")

fig.suptitle("Principal Component Analysis")
fig.tight_layout()

plt.show()