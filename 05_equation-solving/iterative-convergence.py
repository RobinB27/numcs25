import numpy as np
import matplotlib.pyplot as plt

"""
    Convergence of iterative methods shown on 2 examples:
        Linear convergence towards 0.5 using trig. relation
        Approximation of a square root
"""

def linear_convergence(x: float, n: int):
    """ Approximate x towards 0.5 using a trig. relation """
    y = []
    for k in range(n):  
        x += ( np.cos(x)+1 ) / np.sin(x)
        y += [x]
    
    err = np.abs( np.array(y) - x)
    rate = err[1:] / err[:-1]       # Approximate convergence rate
    
    return err, rate


def approx_sqrt(a, x):
    """ Approximate x towards square root of a"""
    exact_sqrt = np.sqrt(a)
    vals = [x]
    x_old = -1
    
    while x_old != x:
        x_old = x
        x = 0.5 * (x + a/x)
        vals += [x]
    
    err = np.abs( np.array(vals) - exact_sqrt )
    return err    
    

# Parameters to set

n = 20
x_lin = [0.2, 0.5, 1, 10, 100, 1000, 10000, 100000]

m = 20
a = 2 
x_sqrt = [1.0, 1.4, 2.0, 10, 50, 100, 1000, 10000, 100000]

# Experiment Runner

err_lin = np.zeros((len(x_lin), n))
for i in range(len(x_lin)):
    err_i, rate = linear_convergence(x_lin[i], n)
    err_lin[i, :] = err_i
    print(f"Convergence rate for x={x_lin[i]}:\t", rate)

err_sqrt = []
for i in range(len(x_sqrt)):
    err_sqrt += [ approx_sqrt(a, x_sqrt[i]) ]

# Plotting

fig, axs = plt.subplots(1, 2)

for i in range(len(x_lin)):
    axs[0].semilogy(np.arange(1, n+1), err_lin[i, :], label=f"x={x_lin[i]}")
    axs[0].scatter(np.arange(1, n+1), err_lin[i, :])

axs[0].set_title("Linear conv. example")
axs[0].legend()
axs[0].grid()

for i in range(len(x_sqrt)):
    axs[1].semilogy(np.arange(1, len(err_sqrt[i])+1), err_sqrt[i], label=f"x={x_sqrt[i]}")
    axs[1].scatter(np.arange(1,len(err_sqrt[i])+1), err_sqrt[i])

axs[1].set_title("Square Root approx. convergence")
axs[1].legend()
axs[1].grid()

fig.tight_layout()

plt.show()