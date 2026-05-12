import os
import sys
from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

import numpy as np
import sklearn

site_pkgs = os.path.dirname(os.path.dirname(sklearn.__file__))
extra_compile_args = ["/O2"] if sys.platform == "win32" else ["-O2"]

src = "asymtree/_splitter.pyx" if USE_CYTHON else "asymtree/_splitter.c"

ext = Extension(
    "asymtree._splitter",
    sources=[src],
    include_dirs=[np.get_include(), site_pkgs],
    extra_compile_args=extra_compile_args,
)

setup(
    ext_modules=cythonize([ext], language_level=3) if USE_CYTHON else [ext],
)
