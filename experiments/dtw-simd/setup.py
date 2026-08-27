"""Build: python setup.py build_ext --inplace

-march=native here because this is a LOCAL experiment; a distributable build
must use -march=x86-64-v3 instead (the -march=native wheel SIGILL on weaker
CI machines is a documented past incident -- see tribble-fis#124).
"""

import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [
            Extension(
                "dtw_simd",
                ["dtw_simd.pyx"],
                extra_compile_args=["-O3", "-march=native", "-fopenmp"],
                extra_link_args=["-fopenmp"],
                include_dirs=[np.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
        ],
        language_level=3,
    ),
)
