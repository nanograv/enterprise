# enterprise_test_data.py

"""This module provides the location of data files for tests as `datadir`.
Currently they are in `tests/data`, and they are based on the 9-yr data release."""

# import this to get the location of the datafiles for tests.  This file
# must be kept in the appropriate location relative to the test data
# dir for this to work.

import os

# Are we on GitHub Actions?
ON_GITHUB = os.getenv("GITHUB_ACTIONS")

# Is libstempo installed?
try:
    import libstempo

    LIBSTEMPO_INSTALLED = True
except ImportError:
    LIBSTEMPO_INSTALLED = False

# Is PINT installed?
try:
    import pint

    PINT_INSTALLED = True
except ImportError:
    PINT_INSTALLED = False

# Location of this file and the test data scripts
testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, "data")
