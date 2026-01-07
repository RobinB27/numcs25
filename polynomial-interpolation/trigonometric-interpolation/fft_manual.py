import numpy as np
import matplotlib.pyplot as plt
import scipy

def root_of_unity(N: int):
    return np.exp( -2.0j*np.pi / N )


def fourier_matrix(N: int):
    F = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            F[i][j] = root_of_unity(N)**(i*j)
    return F


def fourier_coeffs_approx(t: np.ndarray, f, n: int):
    """
        Computes the first n fourier coeffs on the points t using trapezoidal approximation.
        Despite being an approx, this fulfills the interpolation condition.
    """
    N = t.size
    coeffs = np.zeros(n)
    for k in range(n):
        # Approximate instead of using a quadrature
        for l in range(N):
            coeffs[k] += f( t[l] ) * np.exp( -2 * np.pi * k * t[l] )
        coeffs[k] /= N
    return coeffs


def fourier_coeffs_exact(f, n: int):
    """
        Computes the first n fourier coeffs on points t using quadrature.
    """
    coeffs = np.zeros(n)
    for i in range(n):
        # Integrate more precisely via quadrature
        g = lambda x: f(x) * np.exp(-2 * np.pi * i * x)
        print(scipy.integrate.quad(g, 0, 1)[0])
        coeffs[i] = scipy.integrate.quad(g, 0, 1)[0]
    return coeffs


def dft_coeffs(y: np.ndarray):
    """
        Computes DFT coefficients according to definition (slow)
    """
    n = y.size
    coeffs = np.zeros(n)
    omega = root_of_unity(n)
    for k in range(n):
        for j in range(n):
            coeffs[k] += y[j]*omega**(k*j)
    return coeffs
    

def shift_coeffs(coeffs: np.ndarray):
    """
        Shift coeffs (even amount) into correct order for power spectrum
    """
    n = coeffs.size
    z = np.zeros(n)
    z[:n//2] = coeffs[n//2:]
    z[n//2:] = coeffs[:n//2]
    return z


def eval_trigonometric(x: np.ndarray, coeffs: np.ndarray):
    """
        Evaluate polynomial given in trigonometric basis at x
    """
    n = coeffs.size; m = x.size
    y = np.zeros(m, dtype=complex)
    for k in range(n):
        y += coeffs[k] * np.exp(1j * k * x)
    return y
    
    
# Parameters to set

f = lambda x: np.sin(x)
N = 100                     # Points to evaluate on

# Experiment Runner

x = 2 * np.pi * np.arange(N) / N       
coeffs = fourier_matrix(N) @ f(x) / N

# Plotting

M = np.linspace(-5, 5, 1000)

fig, ax = plt.subplots()

ax.plot(M, f(M))
ax.plot(M, eval_trigonometric(M, coeffs))

plt.show()
