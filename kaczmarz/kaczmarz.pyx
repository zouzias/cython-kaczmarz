"""
kaczmarz.pyx

Kaczmarz method based on CSR sparse matrix
"""

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy.linalg import norm


cdef rowNorms(A):
    m = A.shape[0]
    norms = np.zeros(m)
    for i in range(m):
        norms[i] = norm(A.getrow(i).toarray())
        norms[i] = norms[i] * norms[i]
    return norms

cdef c_kaczmarz(A, x, b, iters):
    if not issubclass(type(A), csr_matrix):
        raise ValueError("input matrix must be a scipy sparse matrix of type (CSR)")

    # input matrix is m x n
    m, n = A.shape[0], A.shape[1]

    rNorms = rowNorms(A)
    for k in xrange(1, iters):
        i_k = k % m

        row = A.getrow(i_k)
        dot = row.dot(x)
        val = (b[i_k] - dot) / rNorms[i_k]
        x += val * row
    return x

@cython.boundscheck(False)
@cython.wraparound(False)
def solve(A not None, x not None, b not None, int iters):
    return c_kaczmarz(A, x, b, iters)
