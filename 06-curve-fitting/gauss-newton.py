import numpy as np
import matplotlib.pyplot as plt
import scipy

"""
    For Non-linear problems, the Newton method for
    finding zeros can be used.
"""

def gauss_newton(x: np.ndarray, F, DF, tol=1e-6, maxIter=50):
    """ Gauss-Newton algorithm to solve non-linear problem, needs 'only' the Jacobian """
    s = np.linalg.lstsq(DF(x), F(x))[0] 
    x = x-s
    k = 1
    while np.linalg.norm(s) > tol * np.linalg.norm(x) and k < maxIter:
        s = np.linalg.lstsq(DF(x), F(x))[0]
        x = x-s
        k += 1
    return x, k


def newton(x: np.ndarray, GF, HF, tol=1e1-6, maxIter=50):
    """ Newton requires the Gradient & Hessian instead, more accurate """
    s = np.linalg.solve(HF(x), GF(x))
    x -= s
    k = 1
    while np.linalg.norm(s) > tol * np.linalg.norm(x) and k < maxIter:
        s = np.linalg.solve(HF(x), GF(x))
        x -= s
        k += 1
    return x, k

# Parameters to set

t = np.arange(0, 30, 5); n = len(t)
heat = np.array([24.34, 18.93, 17.09, 16.27, 15.97, 15.91])
cool = np.array([9.66, 18.8, 22.36, 24.07, 24.59, 24.91])

F_1 = lambda a: a[0] + a[1]*np.exp( -a[2]*t ) - heat
F_2 = lambda a: a[0] - a[1]*np.exp( -a[2]*t ) - cool 

def J_1(a):
    J = np.zeros((n, 3))
    for k in range(n):
        J[k, 0] = 1
        J[k, 1] = np.exp( -t[k] * a[2] )
        J[k, 2] = -t[k]*a[1] * np.exp( -t[k]*a[2] )
    return J

def J_2(a):
    J = np.zeros((n, 3))
    for k in range(n):
        J[k, 0] = 1
        J[k, 1] = -np.exp( -t[k]*a[2] )
        J[k, 2] = t[k]*a[1] * np.exp( -t[k]*a[2] )
    return J 

x_1 = np.array([10, 5, 0])
x_2 = np.array([30, 10, 0])

# Experiment Runner

a_1, it_1 = gauss_newton(x_1, F_1, J_1)
a_2, it_2 = gauss_newton(x_2, F_2, J_2)

print(f"f1 took {it_1} Iterations, found: f(x)={a_1[0]} + {a_1[1]}*exp({a_1[2]}*x)")
print(f"f1 took {it_2} Iterations, found: f(x)={a_2[0]} + {a_2[1]}*exp({a_2[2]}*x)")

f_1 = lambda x: a_1[0] + a_1[1]*np.exp( -a_1[2]*x )
f_2 = lambda x: a_2[0] - a_2[1]*np.exp( -a_2[2]*x )

# Plotting

N = np.linspace(0, 25, 1000)
fig, ax = plt.subplots()

ax.plot(N, f_1(N), label="f1")
ax.plot(N, f_2(N), label="f2")
ax.scatter(t, heat, label="Data, f1")
ax.scatter(t, cool, label="Data, f2")

ax.set_title("Gauss-Newton")
ax.grid()
ax.legend()

plt.show()