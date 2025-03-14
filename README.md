# enterprise

![PyPI](https://img.shields.io/pypi/v/enterprise-pulsar)
![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/enterprise-pulsar)
[![Build Status](https://github.com/nanograv/enterprise/workflows/CI-Tests/badge.svg)](https://github.com/nanograv/enterprise/actions)
[![Documentation Status](https://readthedocs.org/projects/enterprise/badge/?version=latest)](https://enterprise.readthedocs.io/en/latest/?badge=latest)
[![Test Coverage](https://codecov.io/gh/nanograv/enterprise/branch/master/graph/badge.svg?token=YXSX3293VF)](https://codecov.io/gh/nanograv/enterprise)
![Python Versions](https://img.shields.io/badge/python-3.8%2C%203.9%2C%203.10%2C%203.11%2C%203.12-blue.svg)

[![Zenodo DOI 4059815](https://zenodo.org/badge/DOI/10.5281/zenodo.4059815.svg)](https://doi.org/10.5281/zenodo.4059815)

ENTERPRISE (Enhanced Numerical Toolbox Enabling a Robust PulsaR
Inference SuitE) is a pulsar timing analysis code, aimed at noise
analysis, gravitational-wave searches, and timing model analysis.

-   Note: `enterprise>=3.0` does not support Python2.7. You must use
    Python \>= 3.8.
-   Free software: MIT license
-   Documentation: <https://enterprise.readthedocs.io>.

## Installation

To install via `pip`, some non-python dependencies are required. See the
[libstempo](https://github.com/vallis/libstempo#pip-install) and
[scikit-sparse](https://github.com/scikit-sparse/scikit-sparse#with-pip)
documentation for more info on how to install these dependencies. Once
these are installed, you can do

```bash
pip install enterprise-pulsar
```

To install via `conda`, simply do

```bash
conda install -c conda-forge enterprise-pulsar
```

### Installing `libstempo` on Apple Silicon (`arm64` architecture)
*`libstempo` is no longer a requirement to install `enterprise`.* However, if installing `libstempo` is desired, it is only available on `osx-64` architectures which is not compatible with `arm64` architectures. To install `libstempo` and `enterprise` on arm64 via `conda`, your new conda environment must be configured correctly. Follow the following commands:
```bash
CONDA_SUBDIR=osx-64 conda create -c conda-forge -n pta_software python=3.12
conda activate pta_software
conda config --env --set subdir osx-64
```

## Attribution

If you make use of this software, please cite it:

    Ellis, J. A., Vallisneri, M., Taylor, S. R., & Baker, P. T. (2020, September 29). ENTERPRISE: Enhanced Numerical Toolbox Enabling a Robust PulsaR Inference SuitE (v3.0.0). Zenodo. http://doi.org/10.5281/zenodo.4059815


    @misc{enterprise,
      author       = {Justin A. Ellis and Michele Vallisneri and Stephen R. Taylor and Paul T. Baker},
      title        = {ENTERPRISE: Enhanced Numerical Toolbox Enabling a Robust PulsaR Inference SuitE},
      month        = sep,
      year         = 2020,
      howpublished = {Zenodo},
      doi          = {10.5281/zenodo.4059815},
      url          = {https://doi.org/10.5281/zenodo.4059815}
    }

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
