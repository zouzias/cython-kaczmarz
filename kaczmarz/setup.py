#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy
import scipy

extensions = [Extension("kaczmarz",
                         extra_compile_args=['-O3'],
                         sources=["kaczmarz.pyx"],
                         include_dirs=[numpy.get_include()])]
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(extensions)
)
