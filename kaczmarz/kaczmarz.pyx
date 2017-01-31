"""
kaczmarz.pyx

Kaczmarz method based on CSR sparse matrix
"""

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np
import math
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy.linalg import norm
from scipy.linalg.blas import daxpy


cdef rowNorms(A):
    m = A.shape[0]
    norms = np.zeros(m)
    for i in range(m):
        norms[i] = norm(A.getrow(i).toarray())
        norms[i] = norms[i] * norms[i]
    return norms

cdef colNorms(A):
    n = A.shape[1]
    norms = np.zeros(n)
    for j in range(n):
        norms[j] = norm(A.getcol(j).toarray())
        norms[j] = norms[j] * norms[j]
    return norms

cdef c_rek(A, x, b, iters):
    if not issubclass(type(A), csr_matrix):
        raise ValueError("input matrix must be a scipy sparse matrix of type (CSR)")

    # input matrix is m x n
    m, n = A.shape[0], A.shape[1]
    A_t = A.tocsc()

    # Compute squared row and column norms
    rNorms = rowNorms(A)
    cNorms = colNorms(A)

    # Initialize variable z
    z = np.array(b)

    for k in xrange(1, iters):
        i_k = k % m
        j_k = k % n

        row = A.getrow(i_k)
        col = scipy.transpose(A_t.getcol(j_k))

        alpha = - col.dot(z) / cNorms[j_k]
        z += alpha * col

        dot = row.dot(x)
        beta = (b[i_k] - z[i_k] - dot) / rNorms[i_k]
        x += beta * row
    return x

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

@cython.boundscheck(False)
@cython.wraparound(False)
def rek(A not None, x not None, b not None, int iters):
    return c_rek(A, x, b, iters)
