#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy>=1.16.3",
    "scipy>=1.2.0",
    "ephem>=3.7.6.0",
    "healpy>=1.14.0",
    "scikit-sparse>=0.4.5",
    "pint-pulsar>=0.8.2",
    "libstempo>=2.4.0",
]

test_requirements = []


setup(
    name="enterprise-pulsar",
    description="ENTERPRISE (Enhanced Numerical Toolbox Enabling a Robust PulsaR Inference SuitE)",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Justin A. Ellis",
    author_email="justin.ellis18@gmail.com",
    url="https://github.com/nanograv/enterprise",
    packages=["enterprise", "enterprise.signals"],
    package_dir={"enterprise": "enterprise"},
    include_package_data=True,
    package_data={"enterprise": ["datafiles/*", "datafiles/ephemeris/*", "datafiles/ng9/*", "datafiles/mdc_open1/*"]},
    python_requires=">=3.6, <3.10",
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords="enterprise",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    test_suite="tests",
    tests_require=test_requirements,
)
