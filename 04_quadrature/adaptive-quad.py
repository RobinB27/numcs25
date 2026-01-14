import numpy as np
import matplotlib.pyplot as plt
import scipy

"""
    Adaptive Quadrature: pick evaluation points by estimating 
    which regions of the interval contribute more/less to 
    quadrature error. (Simpson quad for reference)
    
    Graph overlays generated grids over the function.
"""

def adaptive_quad(f, M, rtol = 1e-6, atol = 1e-10, results = None, grids = None):
    """ Calculate adaptive quad. of f on given grid M"""
    h = np.diff(M)
    midp = 0.5 * ( M[:-1] + M[1:] )
    fx   = f(M)
    fmid = f(midp)
    
    # Local quadratures to estimate local errors further down
    trap_local = h/4 * ( fx[:-1] + 2*fmid + fx[1:] )
    simp_local = h/6 * ( fx[:-1] + 4*fmid + fx[1:] )
    Q = np.sum(simp_local)
    
    # Save intermediate results for plotting
    if (results is not None): results.append(Q)
    if (grids is not None):   grids.append(M)
    
    # Estimate local & total error
    err_loc = np.abs( simp_local - trap_local )
    err = np.sum(err_loc)
    err_avg = err / len(err_loc)
    
    # Refine grid in high-error regions, if needed
    if (err > rtol*np.abs(Q) and err > atol):
        refcells = np.nonzero( err_loc > 0.9 * err_avg )[0]     # Find problematic cells
        M_new = np.sort( np.append(M, midp[refcells]) )         # Update grid
        Q = adaptive_quad(f, M_new, rtol, atol, results, grids)        # Try again
    
    return Q

# Parameters to set

f = lambda x: 1 / (10**(-4) + x**2)     # Function that profits from adaptive quad.
a, b = -1, 1
n = 100     # Initial grid sub-interval count
N = 1000    # points to plot function on

# Experiment Runner

M_init = np.linspace(a, b, n)
results = []; grids = []
Q_adapt = adaptive_quad(f, M_init, results=results, grids=grids)
Q_precise = scipy.integrate.quad(f, a, b)[0]    # Internally uses an adaptive quad. too

err = np.zeros(len(results))
for i in range(len(results)):
    err[i] = np.abs( results[i] - Q_precise )

# Plotting

fig, axs = plt.subplots(1, 2)

x = np.linspace(a, b, N)
y = f(x)

y_max = np.max( y )
y_scale = np.linspace(y_max/len(grids) , y_max, len(grids))

axs[0].plot(x, y)
for i, M in enumerate(grids):
    axs[0].scatter(grids[i], np.full(len(grids[i]), y_scale[i]), s=0.1, color="steelblue")

axs[0].set_title("Function")
axs[0].grid()
axs[0].set_ylabel("f(x) / grids generated")

x_scale = np.arange(1, len(results)+1)
axs[1].semilogy(x_scale, err, label="Adaptive")
axs[1].scatter(x_scale, err)

axs[1].set_title("Convergence")
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("Error")
axs[1].legend()
axs[1].grid()

fig.tight_layout()

plt.show()