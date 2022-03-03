pychell API reference
=====================

Data Containers
+++++++++++++++

pychell makes use of the following instrument-independent data objects for echellograms, 1-dimensional spectra, and radial velocities.

.. automodule:: pychell.data.spectraldata
    :members:
    :show-inheritance:

.. automodule:: pychell.data.rvdata
    :members:
    :show-inheritance:


Spectrographs
+++++++++++++

pychell currently supports the following tested spectrograph configurations.

#. iSHELL
    Reduction: Kgas mode, optimal
    RV generation: Order by order, Kgas mode, 13 CH4 gas cell.
#. PARVI
    Reduction: optimal and SP
    RV generation: Kgas mode
#. HARPS
    RV generation: Chunked, based on ThAr wavelength solution.
#. ESPRESSO
    RV generation: Chunked, based on comb wavelength solution.
#. CHIRON
    RV generation: Order-by-order, iodine gas cell
#. MINERVA
    RV generation: Order-by-order, iodine gas cell.


Coming Soon
-----------

#. NIRSPEC/KPIC
#. NEID