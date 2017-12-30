from distutils.core import setup, Extension

gentrain = Extension('gentrain',
                    sources = ['gentrain.cpp'])

setup (name = 'openke',
       version = '1.0',
       description = 'Dataset builder helper.',
       ext_modules = [gentrain])
