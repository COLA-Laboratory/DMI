from setuptools import setup, Extension

# Compile *test.cpp* into a shared library 
setup(
    ext_modules=[Extension('libIDTLZ',
                           ['IDTLZ/IDTLZ.c',
                            'IDTLZ/IDTLZ1.c',
                            'IDTLZ/IDTLZ2.c',
                            'IDTLZ/IDTLZ3.c',
                            'IDTLZ/IDTLZ4.c'],
                           headers=['IDTLZ/IDTLZ.h']),],
)

