import numpy as np
import sympy as sp

"""
    Sympy caluclates integrals / derivatives symbolically,
    so they can be used as 'analytical' reference values
    for numeric simulations. 
    
    It can also automatically build structures like the Jacobian, Gradient, Hessian etc.
"""

F = lambda x: np.array([
    np.exp(x[0]*x[1]) + x[0]**2 + x[1] - 6/5,
    x[0]**2 + x[1]**2 + x[0] - 11/20
]) 

DF = lambda x: np.array([
    [ x[1]*np.exp(x[0]*x[1]) + 2*x[0],  x[0]*np.exp(x[0]*x[1]) + 1 ],
    [ 2*x[0] + 1,                       2*x[1] ]
])

a, b = (0, 1)

x, y = sp.symbols("x, y")
f1 = sp.exp(x*y) + x**2 + y - 6/5
f2 = x**2 + y**2 + x - 11/20

# Differentiate separately

df1x = sp.diff(f1, x)
df1y = sp.diff(f1, y)

print(df1x, "\n", df1y)

# Or define as sympy 2d function directly, and get Jacobian via sympy

f = sp.Matrix([
    [sp.exp(x*y) + x**2 + y - 6/5],
    [x**2 + y**2 + x - 11/20]
])
X  = sp.Matrix([x, y])      # Values to differentiate

jacobi = f.jacobian(X)
lambda_DF = sp.lambdify([x, y], jacobi)

print("Jacobian\n", jacobi)
print("Sympy Jacobi on (1, 1):\n", lambda_DF(1, 1), "\n", "Numpy Jacobi on (1, 1):\n", DF(np.array([1, 1])))

# Hessian & Gradient

g = x**2 + y**2 + 4/7

grad = [ sp.diff(f, k) for k in [x, y] ]
hess = [ [sp.diff(f, k).diff(j) for k in [x, y]] for j in [x, y] ]

print("Gradient:\n", grad)
print("Hessian:\n", hess)

# Roots

g1 = 2*x**2 + 4*x - 12/7
zeros = sp.roots(g1)

print("Zeros:\t", zeros)

# Chebyshev polynomials

deg = 12
Tpoly = sp.chebyshevt_poly(12)
Upoly = sp.chebyshevu_poly(12)

print(f"Chebyshev poly deg {deg}:\n", Tpoly, "\n", Upoly)
