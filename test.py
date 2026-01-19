import numpy as np;
import matplotlib.pyplot as plt;

# Gauss Quad Task

num = - np.sqrt( 3/7 + 2/7*np.sqrt( 6/5 ) )
print(num)

def gaussquad(n):
    """
    6 Compute nodes and weights for Gauss-Legendre quadrature.
    """
    i = np.arange(1,n) 
    b = i / np.sqrt(4*i**2 - 1)
    J = np.diag(b, -1) + np.diag(b, 1) 
    x, ev = np.linalg.eigh(J)
    w = 2 * ev[0,:]**2
    return x, w

print("Gauss Quad:\t", gaussquad(4)[1])

fig, ax = plt.subplots(1, 2)
a, b = 0, 1
N = 1000
vals = np.linspace(a, b, N)

# Plotting, 1

f1 = lambda x: np.exp(-x**2)
f2 = lambda x: 1 + np.power(x, 2/5)

# Plotting, 2

q1 = lambda x: 1/(2+np.sin(6*x*np.pi))
q2 = lambda x: x
q3 = lambda x: np.sin(np.pi * x)
q4 = lambda x: 1/(2+np.sin(7*x))
q5 = lambda x: np.sin(10*np.pi*x)

q = [q1, q2, q3, q4, q5]

ax[0].plot(vals, f1(vals))
ax[0].plot(vals, f2(vals))

for i, qi in enumerate(q):
    ax[1].plot(vals, qi(vals), label=f"{i}")
ax[1].legend()
ax[1].grid()

# Linear system task

F = lambda x: np.array([
    np.exp(x[0]*x[1]) + x[0]**2 + x[1] - 6/5,
    x[0]**2 + x[1]**2 + x[0] - 11/20
]) 
DF = lambda x: np.array([
    [ x[1]*np.exp(x[0]*x[1]) + 2*x[0],  x[0]*np.exp(x[0]*x[1]) + 1 ],
    [ 2*x[0] + 1,                       2*x[1] ]
])

def newton(x, F, DF, tol=1e-12, maxit=50, arr=None):
    # Newton Iteration
    for i in range(maxit):
        s = np.linalg.solve(DF(x), F(x))
        x -= s
        if np.linalg.norm(s) < tol*np.linalg.norm(x):
            return x

x0 = np.array([0.1, 0.1])
sol = newton(x0, F, DF)
print("Lin system solution:\t", round(sol[0], 4), round(sol[1], 4))

# Newton task

x_vals = np.array([
    0, 1/3, 2/3, 1
])

p = lambda x: 1 + 2*x - 3*x**2 + 4*x**3
px = p(x_vals)
print("Poly at x:\t",px)

def divdiff(x, y):
    n = y.size
    T = np.zeros((n, n))
    T[:,0] = y
    for level in range(1, n):
        for i in range(n-level):
            T[i, level] = (T[i+1,level-1] - T[i,level-1]) / (x[i+level]-x[i])
    return T[0,:]

sol_divdiff = divdiff(x_vals, px)
print("divdiff:\t", sol_divdiff)

plt.show()