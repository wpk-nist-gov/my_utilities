from distutils.core import setup
from Cython.Build import cythonize
import numpy


#python setup.py build_ext --inplace
# setup(
#     name = 'distance module',
#     include_path = [numpy.get_include()],
#     ext_modules = cythonize("mydist.pyx"),
# )

setup(
    ext_modules=cythonize("distcalc.pyx"),
    include_dirs=[numpy.get_include()]
)    
