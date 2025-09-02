from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

# Define extensions with proper module names
extensions = [
    Extension("bonnpatch.core.gillepsie", ["core/gillepsie.pyx"]),
    Extension("bonnpatch.core.hohd", ["core/hohd.pyx"]),
    Extension("bonnpatch.core.histograms", ["core/helper_histograms.pyx"]),
    Extension("bonnpatch.core.generation", ["core/helper_generation.py"]),
    Extension("bonnpatch.core.optimization", ["core/helper_optimization.py"]),
]

setup(
    name="bonnpatch",
    version="0.1.0",
    description="BonnPatch provides python implementations in the realm of Patch clamp recordings for ion channel gating.",
    long_description="BonnPatch provides python implementations in the realm of Patch clamp recordings for ion channel gating. Supporting the computation of theoretical densities, an implementation of the Higher Order Hinkley Detector (HOHD) and fast histogram computations.",
    long_description_content_type="text/markdown",
    author="Manuel Hinz",
    author_email="mh@mssh.dev",
    url="https://github.com/mathmanuelhinz/bonnpatch",
    packages=find_packages(),  # Automatically finds all packages including core and future packages
    ext_modules=cythonize(extensions, compiler_directives={'language_level': 3}),
    include_dirs=[numpy.get_include()],
    install_requires=[
        "numpy>=1.18.0",
        "cython>=0.29.0"
    ],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    zip_safe=False,  # Important for Cython extensions
)