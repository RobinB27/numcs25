import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_solve, lu_factor, norm

"""
    Newton convergence is strongly dependent on initial guess.
    The method often overshoots, so dampening the change using heuristics is useful.
    (However, still doesn't always converge)
    
    This works roughly as follows:
        - Calculate the proper newton step
        - Find good dampening factor via simplified step (reuse Jacobian): s_damp
        - Apply good dampening factor with proper newton step:             s
"""

def dampened_newton(x: np.ndarray, F, DF, q=0.5, rtol=1e-10, atol=1e-12, res_x=None, res_damp=None, res_fx=None):
    """ Dampened Newton with dampening factor q """

    if res_x is not None: res_x.append(x.copy())
    if res_fx is not None: res_fx.append(F(x))
    if res_damp is not None: res_damp.append(1)

    lup    = lu_factor(DF(x))         # LU factorization for efficiency, can also solve directly
    s      = lu_solve(lup, F(x))      # 1st proper Newton correction
    damp   = 1                        # Start with no dampening
    x_damp = x - damp*s    
    s_damp = lu_solve(lup, F(x_damp)) # 1st simplified Newton correction (Reusing previous Jacobian)
    
    if res_x is not None: res_x.append(x_damp.copy())
    if res_fx is not None: res_fx.append(F(x_damp))
    if res_damp is not None: res_damp.append(damp)

    
    while norm(s_damp) > rtol * norm(x_damp) and norm(s_damp) > atol:
        while norm(s_damp) > (1-damp*q) * norm(s):  # Reduce dampening while step is still too aggresive
            damp *= q
            if damp < 1e-4: return x                # Conclude dampening doesn't work anymore
            x_damp = x - damp*s                     # Try weaker dampening instead
            s_damp = lu_solve(lup, F(x_damp))       # Here we reuse the Jacobian
        
        x = x_damp     # Accept this dampened iteration, continue with next proper step
        
        if res_x is not None: res_x.append(x_damp)
        if res_fx is not None: res_fx.append(F(x_damp))
        if res_damp is not None: res_damp.append(damp)
        
        lup    = lu_factor(DF(x))            # Update Jacobian
        s      = lu_solve(lup, F(x))         # Next proper Newton correction
        damp   = min( damp/q, 1 )
        x_damp = x - damp*s
        s_damp = lu_solve(lup, F(x_damp))    # Next simplified Newton correction
    
    return x_damp

# Parameters to set

x_exact = 0
x_start = np.array([20])
F       = lambda x: np.array([ np.arctan(x[0]) ])   # Accepts multivariate functions too
DF      = lambda x: np.array([ 1 / (1+x[0]**2) ])
q       = 0.5   # How dast dampening is reduced

a, b = -25, 25  # Plot bounds
N = 1000        # Plot density

# Experiment Runner

res_x = []; res_fx = []; res_damp = []
x = dampened_newton(x_start, F, DF, q, res_x=res_x, res_damp=res_damp, res_fx=res_fx)
err = np.abs( np.linalg.norm(x_exact) - np.linalg.norm(x) )

errs = np.zeros(len(res_x))
for i in range(len(res_x)): errs[i] = np.abs( np.linalg.norm(x_exact) - np.linalg.norm(res_x[i]))

print("Starting point used:\t", x_start)
print("Solution (Damp. Newt.):\t", x)
print("Exact Solution:\t", x_exact)
print("Error:\t", err)

args = np.linspace(a, b, N)
vals = np.zeros(N)
for i, arg in enumerate(args): vals[i] = F(np.array([arg])) 

# Plotting

fig, axs = plt.subplots(1, 3)

axs[0].plot(args, vals, label="f(x)")
axs[0].scatter(res_x, res_fx, label="Newton")
axs[0].scatter(res_x[0], res_fx[0], label="Start")
axs[0].scatter(res_x[-1], res_fx[-1], label=f"End, k={len(res_x)}")
axs[0].set_xlim(a, b)
axs[0].hlines(0, a, b)

axs[0].set_title("Dampened Newton")
axs[0].legend()
axs[0].grid()

axs[1].semilogy(np.arange(1, len(errs)+1), errs)

axs[1].set_title("Convergence")
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("Error")
axs[1].grid()

axs[2].semilogy(np.arange(1, len(res_damp)+1), res_damp)

axs[2].set_title("Dampening value")
axs[2].set_ylabel("Lambda")
axs[2].set_xlabel("Iterations")
axs[2].grid()

fig.tight_layout()

plt.show()