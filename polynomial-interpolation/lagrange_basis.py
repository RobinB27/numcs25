import numpy as np
import matplotlib.pyplot as plt

def weights_barycentric_seq(x: np.ndarray):
    """
        All x should be pairwise distinct.
    """
    n = x.size
    w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            w[i] = w[i] * 1/(x[i] - x[j])
    return w
    

def weights_barycentric_vec(x: np.ndarray):
    """
        All x should be pairwise distinct.
    """
    n = x.size
    w = np.zeros(n)
    for i in range(n):
        w[i] = 1/( np.prod(x[i] - x[0:i]) * np.prod(x[i] - x[i+1:n]) )
    return w


def test_weights_barycentric(x: np.ndarray):
    """
        Test both versions. Output should be roughly the same
    """
    seq = weights_barycentric_seq(x)
    vec = weights_barycentric_vec(x)
    print("max err found:\n", np.max( np.abs(seq - vec)))


def weights_barycentric(x: np.ndarray):
    """
        Vectorized approach is faster
    """
    return weights_barycentric_vec(x)


def eval_barycentric(w: np.ndarray, data: np.ndarray, y: np.ndarray, x: np.ndarray):
    """
        Sequentially, vectorizing the bary. formula is difficult
    """
    n = x.size
    tmp = np.ones(n)
    for i in range(n): tmp[i] = eval_barycentric_scalar(w, data, y, x[i])
    return tmp
    
    
def eval_barycentric_scalar(w: np.ndarray, data: np.ndarray, y: np.ndarray, x):
    """
        Barycentric interpolation formula for a single value x
    """
    n = data.size;
    bottom = np.sum( w / (x - data) )
    top = np.sum( w / (x - data) * y)
    return top / bottom


def chebyshev_abszissa(n: int, a: float, b: float):
    """
        Returns n+1 chebyshev abszissa on [a,b]
    """
    tmp = np.zeros(n+1)
    for k in range(n+1):
        tmp[k] = a + (1/2)*(b-a) * (np.cos( (k * np.pi) / n ) + 1)
    return tmp

    
# Parameters to set

points_to_try = 60
chebyshev_abszissa_count = 60
deg_polyfit = 60
#f = lambda x: 1/(1 + 5*x**2)    # Runge function
f = lambda x: x**8 + (1 + 2**20) * x**4 + 1     # From a hw task

# Experiment runner

x = np.linspace(-1, 1, points_to_try, endpoint=True);   # equidistant points
y = f(x)
w = weights_barycentric(x)

test_weights_barycentric(x)

x_cheb = chebyshev_abszissa(chebyshev_abszissa_count, -1, 1)
y_cheb = f(x_cheb)
w_cheb = weights_barycentric(x_cheb)

# plotting, functions

n = np.linspace(-1, 1, 1000)
fig, axs = plt.subplots(1, 2)

axs[0].plot(n, f(n), label="f", color="gray")
axs[0].plot(n, eval_barycentric(w, x, y, n), label="p (equidistant)", color="blue")
axs[0].plot(n, eval_barycentric(w_cheb, x_cheb, y_cheb, n), label="p (chebychev)", color="red")
axs[0].plot(n, np.polyval(np.polyfit(x, y, deg_polyfit), n), label="p (polyfit)", color="orange")
axs[0].scatter(x, y, label="x", facecolor="blue", edgecolors="black")
axs[0].scatter(x_cheb, y_cheb, label="x (chebychev)", facecolor="red", edgecolor="black")

axs[0].legend()
axs[0].set_title("Lagrange basis approx.")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].grid(True)

# plotting, errors

axs[1].plot(n, np.abs(f(n) - eval_barycentric(w, x, y, n)), label="p (equidistant)", color="blue")
axs[1].plot(n, np.abs(f(n) - eval_barycentric(w_cheb, x_cheb, y_cheb, n)), label="p (chebyshev)", color="red")
axs[1].plot(n, np.abs(f(n) - np.polyval(np.polyfit(x, y, deg_polyfit), n)), label="p (polyfit)", color="orange")

axs[1].legend()
axs[1].set_title("Error")
axs[1].set_yscale('log')
axs[1].set_xlabel("x")
axs[1].set_ylabel("abs(f(x) - p(x))")
axs[1].grid(True)

plt.show()
