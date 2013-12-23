# -*- coding: utf-8 -*-
import os
from setuptools import setup
version="0.03"
README = os.path.join(os.path.dirname(__file__),"README.rst")
long_description = open(README).read() + '\n\n'
setup(name="changefinder",
      version=version,
      description = ('Online Change-Point Detection Library based on ChangeFinder Algorithm.'),
      long_description = long_description,
      classifiers = ["Programming Language :: Python","Topic :: Scientific/Engineering"],
      keywords = 'scipy, numpy, timeseries analysis',
      author = 'Shunsuke Aihara',
      author_email = "s.aihara gmail com",
      url = 'https://bitbucket.org/aihara/changefinder/',
      license = "MIT License",
      packages = ["changefinder"],
      install_requires=["numpy","scipy","statsmodels","nose"],
      test_suite='nose.collector',
      tests_require=['Nose','numpy','scipy','statsmodels'],
      )
