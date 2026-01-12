import numpy as np
import matplotlib.pyplot as plt

"""
    Deriving trigonomertic polynomials only requires changing the coefficients.
    Thus, trig. polys can easily approximate a function's derivatives 
"""

def zero_pad(v: np.ndarray, N: int):
    """Apply zero-padding to size N to a vector v """
    n = v.size
    if (N < n): raise ValueError(f"ERROR: Zeropadding for N smaller than vector length: {N} < {n}")
    
    u = np.zeros(N, dtype=complex)
    u[:n//2] = v[:n//2]
    u[N-n//2:] = v[n//2:]
    return u


def eval_trig_poly(y: np.ndarray, N: int):
    """ Evaluate trig poly generated using y on N points """
    n = y.size
    if (n % 2 != 0): raise ValueError(f"ERROR: y must be of even length, len(y)={n}")
    
    coeffs = np.fft.fft(y) * 1/n
    coeffs = zero_pad(coeffs, N)
    return np.fft.ifft(coeffs) * N


def eval_trig_poly_d1(y: np.ndarray, N: int):
    """ Evaluates first der. of trig poly generated using y on N points """
    n = y.size
    if (n % 2 != 0): raise ValueError(f"ERROR: y must be of even length, len(y)={n}")
    
    coeffs = np.fft.fft(y) * 1/n
    
    for i in range(0, n//2):
        coeffs[i] *= (2.0j * np.pi * i)
    for i in range(n//2, n):
        coeffs[i] *= (2.0j * np.pi * (i - n))
    
    coeffs = zero_pad(coeffs, N)
    return np.fft.ifft(coeffs) * N


def convergence_analysis(a: float, b: float, f, df, n_eval: int, n_conv: int):
    """
        Evaluate convergence of a trig poly towards f or the derivative df
        n_eval = amount of points in the interval to use
        n_conv = Highest power of 2 to check convergence for
    """
    x = np.linspace(a, b, n_eval, endpoint=False)
    fx = f(x)
    dfx = df(x)
    
    norms_f = np.zeros(n_conv)
    norms_df = np.zeros(n_conv)
    
    N = 2 ** np.arange(1, n_conv+1)
    for i, n in enumerate(N):
        eval_points = np.linspace(a, b, n, endpoint=False)
        y = f(eval_points)
        
        values_f = np.real(eval_trig_poly(y, n_eval))
        values_df = np.real(eval_trig_poly_d1(y, n_eval)) 
        
        norms_f[i] = np.linalg.norm(values_f-fx) / np.sqrt(n_eval)
        norms_df[i] = np.linalg.norm(values_df-dfx) / np.sqrt(n_eval)

    return norms_f, norms_df
        
# Parameters to set

f = lambda x: np.sin(2*np.pi*x)
df = lambda x: (2*np.pi) * np.cos(2*np.pi*x)
a, b = 0, 1         # eval_trig_poly_d1 needs to be updated to handle other intervals
n_eval = 10000
n_conv = 12

# Experiment runner

x = np.linspace(a, b, n_eval)
y = f(x)
dy = df(x)

norms_f, norms_df = convergence_analysis(a, b, f, df, n_eval, n_conv)
t = 2**np.arange(1, n_conv+1)

# Plotting

figs, axs = plt.subplots(1, 3)

axs[0].plot(x, y, label="f(x)")
axs[0].plot(x, eval_trig_poly(y, n_eval), label="p(x) (Fourier)")

axs[0].set_title("Functions")
axs[0].grid()
axs[0].legend()

axs[1].plot(x, dy, label="f'(x)")
axs[1].plot(x, eval_trig_poly_d1(y, n_eval), label="p'(x) (Fourier)")

axs[1].set_title("Derivatives")
axs[1].grid()
axs[1].legend()

axs[2].semilogy(t, norms_f, label="Error (f)")
axs[2].semilogy(t, norms_df, label="Error (f')")

axs[2].set_title("Convergence rate")
axs[2].grid()
axs[2].legend()

plt.show()