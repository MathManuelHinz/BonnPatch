

## Installation
Create a virtual environment and run 
```{bash}
pip install setuptools cython numpy
python setup.py sdist bdist_wheel
pip install dist/bonnpatch-0.1.0-*.whl
```
to build a wheel and install BonnPatch.

Alternatively run
```{bash}
pip install -e .
```
