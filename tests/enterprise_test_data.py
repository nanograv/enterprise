# enterprise_test_data.py

"""This module provides the location of data files for tests as `datadir`.
Currently they are in `tests/data`, and they are based on the 9-yr data release."""

# import this to get the location of the datafiles for tests.  This file
# must be kept in the appropriate location relative to the test data
# dir for this to work.

import os

# Location of this file and the test data scripts
testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, "data")
