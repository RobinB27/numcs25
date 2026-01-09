import numpy as np
import matplotlib.pyplot as plt
import scipy

"""
    Different ways to do the fourier transform: Matrix, manually, numpy (fft)
"""


def root_of_unity(N: int):
    return np.exp( -2.0j*np.pi / N )


def fourier_matrix(N: int):
    F = np.zeros((N, N), dtype=complex)
    for j in range(N):
        for k in range(N):
            F[j][k] = root_of_unity(N)**(j*k)
    return F


def dft_coeffs(y: np.ndarray):
    """
        Computes DFT coefficients according to definition (slow)
    """
    n = y.size
    coeffs = np.zeros(n, dtype=complex)
    omega = root_of_unity(n)
    for j in range(n):
        for k in range(n):
            coeffs[k] += y[j]*omega**(j*k)
    return coeffs


def fourier_shift(v: np.ndarray):
    """
        Shift the 0-component to the middle, like np.fft.fftshift
    """
    n = len(v)
    mid = (n+1)//2
    return np.concatenate((v[mid:], v[:mid]))


def test_fourier_shift():
    arr = np.array([1, 2, 3, 4, 5, 6])
    print(arr)
    arr = fourier_shift(arr)
    print(arr)


def eval_trigonometric(x: np.ndarray, coeffs: np.ndarray):
    """
        Evaluate polynomial given in trigonometric basis at x
    """
    n = coeffs.size; m = x.size
    k = np.arange(-n//2, n//2)      # Shifted indices
    y = np.zeros(m, dtype=complex)
    for i in range(n):
        y += coeffs[i] * np.exp(1j * k[i] * x)
    return y
    
    
# Parameters to set

f = lambda x: np.sin(2*np.pi*x)      # sin(x), period fitted onto [0,1]
N = 1000                             # Points to evaluate on

# Experiment Runner

test_fourier_shift()

x = np.linspace(0, 2*np.pi, N) 
y = f(x)   
c_matrix = fourier_shift( fourier_matrix(N) @ y ) / N
c_dft    = fourier_shift( dft_coeffs(y) )         / N
c_fft    = np.fft.fftshift( np.fft.fft(y) )       / N

# Plotting

M = np.linspace(0, 1, 1000)

fig, ax = plt.subplots()

ax.plot(M, f(M), label="f(x)")
ax.plot(M, eval_trigonometric(M, c_matrix), label="p(x) (F Matrix)")
ax.plot(M, eval_trigonometric(M, c_dft), label="p(x) (DFT)")
ax.plot(M, eval_trigonometric(M, c_fft), label="p(x) (FFT)")

ax.set_title("Fourier Approximation, variants")
ax.grid()
ax.legend()

plt.show()
