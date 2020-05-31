.. _overview:

Overview
********

Reduction
=========

As of now, reduction can be performed on well-behaved spectrographs with an assumed a single trace per echelle order.  Orders are traced with a density clustering algorithm (sklearn.cluster.DBSCAN). Flats are grouped together according to their "density" (metric ~ angular separation + time separation). Objects are mapped to a master flat according to the closest flat in space and time.

Calibration
+++++++++++

Flat, bias, and dark calibration is performed when provided. Wavelength calibartion via ThAr lamps or LFC's are not currently provided.

Extraction
++++++++++

The trace profile (seeing profile) is estimated by rectifying the order and taking a median crunch in the spectral direction on a high resolution grid. The background sky, *sky(x)* is computed by considering regions of low flux within the trace profile. The profile is then interpolated back into 2d space according to the order locations, *y(x)*. An optimal extraction is iteratively performed on the non-rectified data. Depending on the nature of the user's work, this may not be suitable and one should rely on using an instrument specific reduction package.

Tested Support Status:

1. iSHELL (Kgas, K2, J2 modes, empirical and flat field order tracing)
2. CHIRON (highres mode, R~136k, *under development*)
3. NIRSPEC (K band, *under development*)
4. Generic (single trace per order, minimal support)

Radial Velocities
=================

Computes radial velocities from reduced echelle spectra by forward modeling the individual orders. Only certain instruments are supported, however adding support for a new instrument is relatively straightforward (see below).

Tested Support Status:

1. iSHELL (Kgas mode, methane gas cell)
2. CHIRON (highres mode, R~136k, iodine gas cell)
3. Minerva-Australis (ThAr Lamp calibrated, soon iodine gas cell)
4. NIRSPEC (K band, *under development*)