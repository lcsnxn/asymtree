import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sklearn

site_pkgs = os.path.dirname(os.path.dirname(sklearn.__file__))

extra_compile_args = ["/O2"] if sys.platform == "win32" else ["-O2"]

ext = Extension(
    "threshold_trees._splitter",
    sources=["threshold_trees/_splitter.pyx"],
    include_dirs=[np.get_include(), site_pkgs],
    extra_compile_args=extra_compile_args,
)

setup(
    ext_modules=cythonize([ext], language_level=3),
)
