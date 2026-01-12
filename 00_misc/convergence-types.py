import numpy as np
import matplotlib.pyplot as plt

"""
    Visualisation of linear / exp. convergence for different p, q values
"""

# Parameters to set

N = 20
p_vals = [0.5, 0.9, 1, 2, 5, 7]
q_vals = [0.1, 0.25, 0.5, 0.75, 0.9]

# Experiment Runner

algebraic_conv      = lambda n, p: 1 / (n**p)
exponential_conv    = lambda n, q: q**n 

n_vals = np.arange(1, N+1, 1)

# Plotting

fig, ax = plt.subplots()

for p in p_vals:
    ax.semilogy(n_vals, algebraic_conv(n_vals, p), label=f"p={p}", color="blue")
for q in q_vals:
    ax.semilogy(n_vals, exponential_conv(n_vals, q), label=f"q:{q}", color="orange")

ax.set_title("Convergence Study")
ax.legend()
ax.grid()

plt.show()