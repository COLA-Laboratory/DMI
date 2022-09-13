from setuptools import setup, Extension

# Compile *test.cpp* into a shared library 
setup(
    ext_modules=[Extension('libWFG', ['WFG/WFG.c',
                                      'WFG/WFG1.c',
                                      'WFG/WFG2.c',
                                      'WFG/WFG3.c',
                                      'WFG/WFG4.c',
                                      'WFG/WFG5.c',
                                      'WFG/WFG6.c',
                                      'WFG/WFG7.c',
                                      'WFG/WFG8.c',
                                      'WFG/WFG9.c',
                                      'WFG/WFG4x.c',
                                      'WFG/WFG2x.c'],
                           headers=['WFG/WFG.h'])],
)

