#!/usr/bin/env python

import kaczmarz
import math
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
import numpy as np
from scipy.sparse import rand
from scipy.linalg import norm

def testKaczmarz(m, n, iters):
    np.random.seed(0)
    A = rand(m, n, density=0.9, format="csr")
    xopt = np.random.rand(n)
    b = A.dot(xopt)
    x = np.zeros(n, dtype=np.double)
    x_approx = kaczmarz.solve(A, x, b, iters)
    printError(xopt, x_approx)

def testREK(m, n, iters):
    np.random.seed(0)
    A = rand(m, n, density=0.9, format="csr")
    xopt = np.random.rand(n)
    b = A.dot(xopt)
    x = np.zeros(n, dtype=np.double)
    x_approx = kaczmarz.rek(A, x, b, iters)
    printError(xopt, x_approx)

def printError(xopt, x_approx):
    print("norm(xopt)={}".format(norm(xopt)))
    print("norm(x_approx)={}".format(norm(x_approx)))
    print("Error is {}".format(norm(x_approx - xopt)))


print("Testing Kacmzarz")
testKaczmarz(1000, 100, 10000)

print("Testing REK")
testREK(100, 10, 10000)
