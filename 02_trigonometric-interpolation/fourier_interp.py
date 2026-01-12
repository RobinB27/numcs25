import numpy as np
import matplotlib.pyplot as plt

"""
    Fourier approximation of a function using np.fft
"""

def eval_fourier_poly(x: np.ndarray, coeffs: np.ndarray):
    """
        Evaluate trig. poly. at x using coeffs
    """
    n = x.size; m = coeffs.size
    k = np.arange(-m//2, m//2)      # Shifted indices
    y = np.zeros(n, dtype=complex)
    for i in range(m):
        y += coeffs[i] * np.exp(1j * k[i] * x)
    return y


# Parameters to set

f = lambda x: np.sin(x)
n = 100                 # points in [-1,1] to test for poly
a, b = -5, 5            # Interval to plot on

# Experiment runner

x = np.linspace(0, 2*np.pi, n)
y = f(x)
coeffs = np.fft.fftshift( np.fft.fft(y) ) / n

# Plotting

N = np.linspace(a, b, 1000)

fig, ax = plt.subplots()
ax.plot(N, f(N), label="f(x)")
ax.plot(N, eval_fourier_poly(N, coeffs).real, label="p(x) (Fourier)")

ax.set_title("Fourier Approximation")
ax.grid()
ax.legend()

plt.show()