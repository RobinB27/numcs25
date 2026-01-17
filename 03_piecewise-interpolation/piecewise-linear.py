import numpy as np
import matplotlib.pyplot as plt

"""
    Basic Linear interpolation of a function
"""

def slope(x, j, x_data, y):
    """
        Slope on j-th sub-interval, for x falling inside that sub-interval
    """
    h = np.abs(x_data[j] - x_data[j-1])   # size of sub-interval
    return y[j-1]*(x_data[j] - x)/h + y[j]*(x - x_data[j-1])/h


def eval_linear_interp(x: np.ndarray, x_data, y):
    """
        Evaluate linear interpolation on x, using data points (x_data, y)
        via numpy piecewise functions
    """
    n = x_data.size; m = x.size
    
    # much python-specific stuff going on here ...
        
    condlist = [
        (x_data[j-1] <= x) & (x <= x_data[j])       # Need to use bit-wise & to create np arrays (not 'and')
        for j in range(1, n)
    ]
    funclist = [
        lambda x, ind=j: slope(x, ind, x_data, y)   # setting j here directly actually locks lambda to j=1
        for j in range(1, n)
    ]
    
    return np.piecewise(x, condlist, funclist)


# Parameters to set

f = lambda x: np.sin(x)     # Function to interpolate
a, b = -10, 10              # Interval to interpolate on
n = 10                      # Amount of sub-intervals (equal sizes)
N = 1000                    # Points to use for plot

# Experiment Runner

x = np.linspace(a, b, n)
y = f(x)

vals = np.linspace(a, b, N)
fx = f(vals)
fx_approx = eval_linear_interp(vals, x, y)
err = np.abs( fx - fx_approx )

# Plotting

figs, axs = plt.subplots(2, 1)

axs[0].plot(vals, fx, label="f(x)")
axs[0].plot(vals, fx_approx, label="f(x) (Approx.)")
axs[0].scatter(x, y, label="Interp. points")

axs[0].set_title("Linear Interpolation")
axs[0].grid()
axs[0].legend()

axs[1].plot(vals, err, label="abs. error")

axs[1].set_title("Absolute Error")
axs[1].grid()
axs[1].legend()

plt.show()