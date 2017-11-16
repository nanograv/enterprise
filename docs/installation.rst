.. highlight:: shell

============
Installation
============


.. Stable release
.. --------------
..
.. To install enterprise, run this command in your terminal:
..
.. .. code-block:: console
..
..     $ pip install enterprise
..
.. This is the preferred method to install enterprise, as it will always install the most recent stable release.
..
.. If you don't have `pip`_ installed, this `Python installation guide`_ can guide
.. you through the process.
..
.. .. _pip: https://pip.pypa.io
.. .. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for enterprise can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/nanograv/enterprise

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/nanograv/enterprise/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install numpy
    $ pip install -r requirements.txt
    $ pip install git+https://github.com/vallis/libstempo.git --install-option="--with-tempo2=$TEMPO2"
    $ python setup.py install

If you want to run tests or do any other development then also run:

.. code-block:: console

    $ pip install -r requirements_dev.txt

.. _Github repo: https://github.com/nanograv/enterprise
.. _tarball: https://github.com/nanograv/enterprise/tarball/master
