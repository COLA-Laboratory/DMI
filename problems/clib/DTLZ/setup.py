from setuptools import setup, Extension

# Compile *test.cpp* into a shared library 
setup(
    ext_modules=[Extension('libDTLZ', ['DTLZ/DTLZ.c',
                                       'DTLZ/DTLZ1.c',
                                       'DTLZ/DTLZ2.c',
                                       'DTLZ/DTLZ3.c',
                                       'DTLZ/DTLZ4.c',
                                       'DTLZ/DTLZ5.c',
                                       'DTLZ/DTLZ6.c',
                                       'DTLZ/DTLZ7.c',
                                       'DTLZ/DTLZ7x.c'],
                           headers=['DTLZ/DTLZ.h']), ],
)
