with open('README.rst', 'r') as f:
    readme = f.read()
    
from setuptools import setup, Extension
from pyLIA import __version__
import numpy
if __name__ == '__main__':
    setup(name='pyArUco',
        version=__version__,
        author='Lorenzo Capponi','Tommaso Tocci',
        author_email='lorenzocapponi@outlook.it',
        description='Module for Lock-In Analysis',
        url='https://github.com/LolloCappo/pyArUco',
        py_modules=['pyArUco'],
        long_description=readme,
        install_requires = 'numpy'
      )