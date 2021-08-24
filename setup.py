with open('README.rst', 'r') as f:
    readme = f.read()
    
from setuptools import setup, Extension
from pyLIA import __version__
import numpy
if __name__ == '__main__':
    setup(name='pyLIA',
        version=__version__,
        author='Lorenzo Capponi',
        author_email='lorenzocapponi@outlook.it',
        description='Module for Lock-In Analysis',
        url='https://github.com/LolloCappo/pyLIA',
        py_modules=['pyLIA'],
        long_description=readme,
        install_requires = 'numpy'
      )