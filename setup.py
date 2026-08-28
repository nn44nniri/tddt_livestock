from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["funnel/funnel_core.pyx"], language_level=3),
)
