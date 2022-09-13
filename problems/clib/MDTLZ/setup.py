from setuptools import setup, Extension

# Compile *test.cpp* into a shared library 
setup(
    ext_modules=[Extension('libMDTLZ',
                           ['MDTLZ/MDTLZ.c',
                            'MDTLZ/MDTLZ1.c',
                            'MDTLZ/MDTLZ2.c',
                            'MDTLZ/MDTLZ3.c',
                            'MDTLZ/MDTLZ4.c'],
                           headers=['MDTLZ/MDTLZ.h']),],
)

