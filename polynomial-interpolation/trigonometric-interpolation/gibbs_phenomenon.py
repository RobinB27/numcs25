import numpy as np
import matplotlib.pyplot as plt

def exact_fourier(x: np.ndarray, N: int, a: float, b: float):
    """
        Exact fourier series over x for the characteristic function of [a,b] \subset (0,1)
    """
    c = np.pi * (a + b); d = np.pi * (b - a)
    sol = b-a
    for k in range(1, N, 1):
        sol += 2*np.exp(-1.0j * k * c) * np.sin(k * d) * np.exp(2.0j * np.pi * k * x) / (k * np.pi)
    return sol

# Parameters to set

a, b = 0.25, 0.75
n = 100
f = lambda x: np.piecewise(
    x,
    [x > b, x < a],
    [0, 0, 1]               # 3rd func without any condition is default
)

# Plotting

N = np.linspace(0, 1, 1000)

figs, axs = plt.subplots(1, 2)

axs[0].plot(N, f(N), label="f(x)")
axs[0].plot(N, np.real(exact_fourier(N, n, a, b)), label="p(x) (Fourier)")

axs[0].set_title("Gibbs Phenomenon")
axs[0].legend()

axs[1].vlines([0.25, 0.75], 0, 3, linestyle="dashed", color="gray")
axs[1].plot(N, np.abs( f(N) - exact_fourier(N, 1000, a, b) ), label="n=1000")
axs[1].plot(N, np.abs( f(N) - exact_fourier(N, 100, a, b) ), label="n=100")
axs[1].plot(N, np.abs( f(N) - exact_fourier(N, 10, a, b) ), label="n=10")
axs[1].plot(N, np.abs( f(N) - exact_fourier(N, 5, a, b) ), label="n=5")

axs[1].set_title("Error")
axs[1].legend()

plt.show()