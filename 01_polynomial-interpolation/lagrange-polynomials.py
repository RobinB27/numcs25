import numpy as np
import matplotlib.pyplot as plt

"""
    Visualisation of Lagrange polynomials and naive interpolation
"""

def lagrange_poly(x: np.ndarray, i: int):
    """ Returns lambda func for i-th lagrange poly. defined using x """
    return lambda arg: np.array([ np.prod( (arg[j] - x[0:i]) / (x[i] - x[0:i]) ) * np.prod( (arg[j] - x[i+1:]) / (x[i] - x[i+1:]) ) for j in range(len(arg)) ])

def eval_lagrange_poly(x0: np.ndarray, x: np.ndarray, y: np.ndarray):
    """ Evaluate lagrange poly defined on x with weights y on x0 """
    p = np.zeros(len(x0))
    for i in range(len(x)): 
        p += y[i]*lagrange_poly(x, i)(x0)
    return p

# Parameters to set

x0 = [3]            # Where to evaluate 

x = [1, 2, 4, 7]    # Data for Lagrange
y = [3, 4, 8, 1]    # Weights for Lagrange

a, b = -10, 10      # Bounds for plot

# Experiment runner

x = np.array(x); y = np.array(y); x0 = np.array(x0)

lagrange = [ lagrange_poly(x, i) for i in range(len(x)) ]
vals = [ lagrange[i](x0) for i in range(len(lagrange)) ]
print(vals)

fx0 = eval_lagrange_poly(x0, x, y)
print(fx0)

# Plotting

N = np.linspace(a, b, 1000)

fig, axs = plt.subplots(1, 2)

for i in range(len(lagrange)):
    axs[0].plot(N, lagrange[i](N), label=f"L{i}")
axs[0].scatter(x, np.zeros_like(x), label="Data = Zeros")

axs[0].set_title("Lagrange Polynomials")
axs[0].legend()
axs[0].grid()

axs[1].plot(N, eval_lagrange_poly(N, x, y), label="Lagrange approx.")
axs[1].scatter(x, y, label="Data")

axs[1].set_title("Lagrange Interpolation")
axs[1].legend()
axs[1].grid()
    
plt.show()