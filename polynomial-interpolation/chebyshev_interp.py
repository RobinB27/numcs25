import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

def clenshaw(c: np.ndarray, x: np.ndarray):
    """
        Clenshaw backwards recursion to evaluate polynomial using chebyshev coeffs
    """
    n = c.size; m = x.size
    d = np.zeros((m, 3))    # Save vectors [curr, prev1, prev2] as matrix
    
    for i in range(n-1, -1, -1):
        d[:, 2] = d[:, 1]; d[:, 1] = d[:, 0]
        d[:, 0] = c[i] + (2*x)*d[:, 1] - d[:, 2]
    return d[:, 0] - x*d[:, 1]


def chebychev_coeffs(y: np.ndarray):
    """
        Computes chebyshev coeffs using FFT (from lecture Script)
    """
    n = y.size - 1
    
    t = np.arange(0, 2*n + 2)
    z = np.exp( -np.pi * 1.0j * n / (n + 1)*t) * np.hstack([ y, y[::-1] ])
    
    # Solve lin. system for c
    c = fft.ifft(z)
    
    # recover gamma
    t = np.arange(-n, n+2)
    b = np.real( np.exp(0.5j * np.pi / (n + 1) * t)*c)
    
    # recover coeffs
    coeffs = np.hstack([ 
        b[n], 
        2*b[n+1:2*n+1]
    ])
    
    return coeffs


def chebychev_zeros(n: int, a: int, b: int):
    """
        Computes the zeros of the chebyshev polynomial T^{n+1} on [a,b]
    """
    z = np.zeros(n+1)
    for i in range(n+1):
        z[i] = a + (1/2)*(b-a) * ( np.cos((2*i + 1)/(2*(n+1)) * np.pi) + 1)
    return z


def test_chebychev_zeros():
    """
        Comparing against a vectorized version given in lecture
    """
    
    # A vectorized way to get the zeros for [-1,1]:
    # z = np.cos((2 * np.r_[0:n+1]+1) / (n+1) * np.pi/2)
    
    n = 100
    z_loop = chebychev_zeros(n, -1, 1)
    z_vec = np.cos((2 * np.r_[0:n+1]+1) / (n+1) * np.pi/2)
    print("Looped:\n", z_loop)
    print("Vectorized:\n", z_vec)
    print("Difference\n", np.abs(z_loop - z_vec))
    print("Maximum error\n", np.max( np.abs(z_loop - z_vec) ))
    

# Parameters to set

N = 20                           # Points for the polynomial
a, b = -1, 1                     # Interval
n = np.linspace(a, b, 1000)      # points to evaluate for plots
f = lambda x: 1/(1 + 25*x**2)    # Runge function

# Experiment runner, interpolation

x_cheby = chebychev_zeros(N, a, b)    # Zeros of T^(n+1)
y_cheby = f(x_cheby)
coeffs = chebychev_coeffs(y_cheby)

test_chebychev_zeros()

# Experiment runner, error analysis

n_max = 200

err = np.zeros(n_max-1)
for i in range(2, n_max+1):
    x_cheby = chebychev_zeros(i, a, b)
    y_cheby = f(x_cheby)
    coeffs = chebychev_coeffs(y_cheby)
    
    err[i-2] = np.max( np.abs(f(n) - clenshaw(coeffs, n)) )     

# Plotting, interpolation

figs, axs = plt.subplots(1, 2)

axs[0].plot(n, f(n), label="f")
axs[0].plot(n, clenshaw(coeffs, n), label="p (Clenshaw)")
axs[0].scatter(x_cheby, y_cheby, label="Chebyshev Zeros")

axs[0].legend()
axs[0].set_title("Chebyshev Approximation")

# Plotting, error analysis

axs[1].semilogy( list(range(2, n_max+1)), err, label='Error')

axs[1].set_xlabel("Points used for Poly.")
axs[1].set_ylabel('Error')
axs[1].legend()
axs[1].set_title("Error vs. degree")

plt.show()
