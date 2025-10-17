# setup.py
from setuptools import setup, find_packages

setup(
    name='waxa',
    version='0.0.1',
    url='https://github.com/ucsb-amo/wax-ucsb/waxa',
    author='Jared Pagett',
    author_email='pagett.jared@gmail.com',
    packages=find_packages(), # This will find 'my_package'
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'h5py',
        'datetime',
        'time',
        'copy',
        'subprocess',
        'glob',
        'random']
)