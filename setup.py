from setuptools import setup
from Cython.Build import cythonize
import numpy
setup(
    name='Gillepsie',
    ext_modules=cythonize(["gillepsie.pyx","hohd.pyx","helper_histograms.pyx"]),
    include_dirs=[numpy.get_include()]
)