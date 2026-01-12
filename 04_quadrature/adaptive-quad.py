import numpy as np
import scipy

"""
    Adaptive Quadrature: pick evaluation points by estimating 
    which regions of the interval contribute more/less to 
    quadrature error.
"""

def adaptive_quad(f, M, rtol, atol):
    return 0