pychell API reference
=====================

pychell makes use of specific data objects for echellograms, 1-dimensional spectra, and radial velocities which are all instrument independent.

Data Containers
+++++++++++++++

.. automodule:: pychell.data.spectraldata
    :members:
    :show-inheritance:

.. automodule:: pychell.data.rvdata
    :members:
    :show-inheritance:


pychell currently supports the following spectrographs. Only iSHELL, PARVI, and NIRSPEC come with support for spectral extraction. For spectrographs with multiple modes, typically only the ideal mode for precise RVs is supported. pychell is able to support RVs for all spectrographs from 1-dimensional spectra.

Spectrographs
+++++++++++++

CHIRON
------

.. automodule:: pychell.data.chiron
    :members:


ESPRESSO
--------

.. automodule:: pychell.data.espresso
    :members:


HARPS
-----

.. automodule:: pychell.data.harps
    :members:


iSHELL
------

.. automodule:: pychell.data.ishell
    :members:

MINERVA
-------

.. automodule:: pychell.data.minerva
    :members:

NIRSPEC
-------

.. automodule:: pychell.data.nirspec
    :members:


PARVI
-----

.. automodule:: pychell.data.parvi
    :members: