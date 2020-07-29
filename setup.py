# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='ifcseq',
      version='0.1',
      description='Predict single-cell gene expression for imaging flow cytometry data',
      url='http://github.com/nchlis',
      author='Nikolaos Kosmas Chlis',
      author_email='nchlis@isc.tuc.gr',
      license='MIT',
      packages=['ifcseq'],
               install_requires=[
               'numpy',
               'scikit-learn',
               'joblib'
      ])
#      ,zip_safe=False)
