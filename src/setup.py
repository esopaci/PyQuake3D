from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name="ang_dis_strain",
    sources=["ang_dis_strain.pyx", "ang_dis_strain.cpp"],
    language="c++",
    extra_compile_args=["-std=c++11"]
)

setup(
    ext_modules=cythonize(ext),
)
