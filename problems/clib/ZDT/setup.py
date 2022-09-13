import sys
import os
print('Python %s on %s at %s' % (sys.version, sys.platform, os.getcwd()))
sys.path.extend(['./'])

from setuptools import setup, Extension

setup(
    ext_modules=[Extension('libZDT', ['ZDT/ZDT.c',
                                      'ZDT/ZDT1.c',
                                      'ZDT/ZDT2.c',
                                      'ZDT/ZDT3.c',
                                      'ZDT/ZDT4.c',
                                      'ZDT/ZDT6.c',
                                      'ZDT/ZDT3x.c'],
                           headers=['ZDT/ZDT.h']), ],
)
