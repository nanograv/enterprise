#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    # TODO: put package requirements here
]

test_requirements = []


setup(
    name='enterprise',
    version='2.0.0',
    description="ENTERPRISE (Enhanced Numerical Toolbox Enabling a Robust PulsaR Inference SuitE)",
    long_description=readme + '\n\n' + history,
    author="Justin A. Ellis",
    author_email='justin.ellis18@gmail.com',
    url='https://github.com/nanograv/enterprise',
    packages=[
        'enterprise',
        'enterprise.signals'
    ],
    package_dir={'enterprise':
                 'enterprise'},
    include_package_data=True,
    package_data={'enterprise':['datafiles/*',
                                'datafiles/ephemeris/*',
                                'datafiles/ng9/*',
                                'datafiles/mdc_open1/*']},
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='enterprise',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
