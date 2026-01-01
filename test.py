import numpy as np;
import matplotlib as plt;

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

S, V, D = np.linalg.svd(A)