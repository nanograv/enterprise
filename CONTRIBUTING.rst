.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/nanograv/enterprise/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

enterprise could always use more documentation, whether as part of the
official enterprise docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/nanograv/enterprise/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up ``enterprise`` for local development.

Install the dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

``enterprise`` relies on a lot of other software to function.
If you use the Anaconda distribution of Python, you can get all of this software using ``conda``.
First, you install the latest stable version of ``enterprise``, which will come with all of the dependencies.
Then you remove ``enterprise`` leaving everything else intact.
This way you can use your development version of ``enterprise`` instead of the stable version.
We will also need some additional software that is required to run the tests.

Start with an empty virtual environment, in this case called ``ent_dev``::

    $ conda create -n ent_dev -y -c conda-forge python=3.9
    $ conda activate ent_dev

Now install things by running the commands::

    $ conda install -c conda-forge enterprise-pulsar
    $ conda remove enterprise-pulsar --force
    $ conda install -c conda-forge  black=22.3.0 flake8 sphinx_rtd_theme pytest-cov
    $ pip install coverage-conditional-plugin

Without the dependencies you can still edit the source code, but you won't be able to run tests to see if your changes work!


Get the enterprise source code and get to work!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Fork the ``enterprise`` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/enterprise.git
    $ cd enterprise/
    
3. Set `enterprise/master` as upstream remote::
    
    $ git remote add upstream https://github.com/nanograv/enterprise.git
    
   You can then pull changes from the upstream master branch with::
   
    $ git pull upstream master

4. This is how you set up your fork for local development:
    
    .. note:: 
        You will need to have ``tempo2`` and ``suitesparse`` installed before  
        running these commands.

        If you installed the dependencies via conda, you are good to go!

    If you set up a ``conda`` virtual environment with the dependencies already,
    you can add your local fork of ``enterprise`` to it by running::

    $ pip install -e .

    If you manually installed the dependencies, this will make and activate a 
    Python3 virtual env with your local fork of ``enterprise``::
    
    $ make init
    $ source .enterprise/bin/activate  


5. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

6. When you're done making changes, check that your changes pass flake8 and the tests, including testing other Python versions with tox (tox not implemented yet). Also check that any new docs are formatted correctly::

    $ make test
    $ make docs

   To get flake8 and tox, just pip install them into your virtualenv.

7. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring.
3. The pull request should work for all supported versions of Python: 3.8, 3.9, 3.10, and 3.11. You
   can see the progress of the tests in the `Checks` tab of your GitHub pull request.

Tips
----

To run a subset of tests::

    $ python -m unittest tests.test_enterprise
    
To track and checkout another user's branch::

    $ git remote add other-user-username https://github.com/other-user-username/enterprise.git
    $ git fetch other-user-username
    $ git checkout --track -b branch-name other-user-username/branch-name

