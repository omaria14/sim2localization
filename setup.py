#!/usr/bin/env python

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup


d = generate_distutils_setup(
    packages=["utils", "segmentation", "simulated_scans_handler", "classification"],
    package_dir={'': 'scripts'}
)

setup(**d)