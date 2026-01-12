import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

"""
    Derivatives approximated using Cubic Splines, Pchip & Akima (from scipy)
"""

# Parameters to set

f       = lambda x: np.sin(x * np.pi*2)
df      = lambda x: np.pi*2 * np.cos(x * np.pi*2)
ddf     = lambda x: -np.pi**2 * 4 * np.sin(x * np.pi*2)
dddf    = lambda x: -np.pi**3 * 8 * np.cos(x * np.pi*2)

n = 100     # Points for interpolation
N = 1000    # Points for plots

# Experiment Runner

x = np.linspace(0, 1, n)
y = f(x)

cubic_spline = interp.CubicSpline(x, y)     # Implicitly uses "not-a-knot" condition
pchip = interp.PchipInterpolator(x, y)      # Guarantees monotonicity conservation!
akima = interp.Akima1DInterpolator(x, y)

vals = np.linspace(0, 1, N)

# Plotting

figs, axs = plt.subplots(4, 1)

axs[0].plot(vals, f(vals), label="f(x)", color="lime")
axs[0].plot(vals, cubic_spline(vals), label="Cubic Spline", color="lightblue")
axs[0].plot(vals, pchip(vals), label="Pchip", color="steelblue")
axs[0].plot(vals, akima(vals), label="Akima", color="blue")
axs[0].scatter(x, y, label="data")

axs[0].set_title("Function")
axs[0].grid()
axs[0].legend()

axs[1].plot(vals, df(vals), color="lime")
axs[1].plot(vals, cubic_spline(vals, nu=1), label="Cubic Spline", color="lightblue")
axs[1].plot(vals, pchip(vals, nu=1), label="Pchip", color="steelblue")
axs[1].plot(vals, akima(vals, nu=1), label="Akima", color="blue")

axs[1].set_title("Function: 1st derivative")
axs[1].grid()

axs[2].plot(vals, ddf(vals), color="lime")
axs[2].plot(vals, cubic_spline(vals, nu=2), label="Cubic Spline", color="lightblue")
axs[2].plot(vals, pchip(vals, nu=2), label="Pchip", color="steelblue")
axs[2].plot(vals, akima(vals, nu=2), label="Akima", color="blue")

axs[2].set_title("Function: 2nd derivative")
axs[2].grid()

axs[3].plot(vals, dddf(vals), color="lime")
axs[3].plot(vals, cubic_spline(vals, nu=3), label="Cubic Spline", color="lightblue")
axs[3].plot(vals, pchip(vals, nu=3), label="Pchip", color="steelblue")
axs[3].plot(vals, akima(vals, nu=3), label="Akima", color="blue")

axs[3].set_title("Function: 3rd derivative")
axs[3].grid()
lim = 2*np.max(dddf(vals))  # Error is huge, limit y-axis so original func is visible.
axs[3].set_ylim(-lim, lim)

figs.tight_layout()

plt.show()