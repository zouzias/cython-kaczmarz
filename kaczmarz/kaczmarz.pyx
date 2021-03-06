"""
kaczmarz.pyx

Kaczmarz method based on CSR sparse matrix
"""

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy.linalg import norm
from scipy.linalg.blas import daxpy


cdef rowNorms(A):
    m = A.shape[0]
    norms = np.zeros(m)
    for i in xrange(m):
        for l in xrange(A.indptr[i], A.indptr[i + 1]):
            norms[i] += A.data[l]**2
    return norms

cdef colNorms(A):
    m, n = A.shape[0], A.shape[1]
    norms = np.zeros(n)
    for i in xrange(m):
        for l in xrange(A.indptr[i], A.indptr[i + 1]):
            norms[A.indices[l]] += A.data[l]**2

    return norms

cdef c_extendedKaczmarz(A, x, b, iters):
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

    cdef double dot = 0.0
    cdef double alpha = 0.0

    if not issubclass(type(A), csr_matrix):
        raise ValueError("input matrix must be a scipy sparse matrix of type (CSR)")

    # input matrix is m x n
    m, n = A.shape[0], A.shape[1]

    rNorms = rowNorms(A)
    for k in xrange(1, iters):
        i_k = k % m

        row = A.getrow(i_k).toarray()

        dot = 0.0
        for l in xrange(A.indptr[i_k], A.indptr[i_k + 1]):
            dot += A.data[l] * x[A.indices[l]]

        alpha = (b[i_k] - dot) / rNorms[i_k]

        for l in xrange(A.indptr[i_k],A.indptr[i_k + 1]):
            x[A.indices[l]] += alpha * A.data[l]
    return x

@cython.boundscheck(False)
@cython.wraparound(False)
def solve(A not None, x not None, b not None, int iters):
    return c_kaczmarz(A, x, b, iters)

@cython.boundscheck(False)
@cython.wraparound(False)
def rek(A not None, x not None, b not None, int iters):
    return c_extendedKaczmarz(A, x, b, iters)
