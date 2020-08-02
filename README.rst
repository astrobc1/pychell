=======
pychell
=======

The tldr;
=========

A environment to:

1. Reduce single trace multi-order echelle spectra.
2. Generate precise radial velocities for stellar sources.

Install with ``pip install .`` from the head directory.

Only certain instruments are supported, however adding support for a new instrument is relatively straightforward (see extending pychell).

Reduction
=========

As of now, reduction can be performed on well-behaved spectrographs with a single trace per echelle order.

Order Tracing
+++++++++++++

By default, orders are traced with a density clustering algorithm (sklearn.cluster.DBSCAN) on either the flat fields (preferred) or data, but hard-coded maps may be used if the order locations on the detector are known to be relatively stable. If order locations must be uniquely determined from the data, tweaking the dbscan algorithm will likely be necessary. The precise order locations may still be refined for a unique exposure via iteratively cross-correlating an estimated trace profile (seeing profile) with the unrectified 2d image, so estimated order map algorithms are both sufficient and preferred, except in the case of crowded orders.

Calibration
+++++++++++

Flat, bias, and dark calibration are performed when provided and set. Telluric calibration (flat star observations) Wavelength calibartion via ThAr lamps or LFC's are not currently supported, but intended to be in the future if enough desired. Wavelength telluric calibration would be performed post-extraction.

Extraction
++++++++++

The trace profile (seeing profile) is estimated by rectifying the order and taking a median crunch in the spectral direction on a high resolution grid (tuneable parameter). The background sky, *sky(x)* is computed by considering regions of low flux (< 5 percent) within a given column. By default, an optimal extraction is iteratively performed on the non-rectified data, although the trace profile is interpolated for each column via cubic splines from it's pre-defined fiducial grid according to the trace positions. Depending on the nature of the user's work, this *may* not be suitable and one should rely on using an instrument specific reduction package or implementing one's own optimal extraction algorithm(s).

Support Status:

1. iSHELL / IRTF (Kgas, K2, J2 modes via flat field order tracing)
2. CHIRON / SMARTS 1.5 m (highres mode, R~136k, *under development*)
3. NIRSPEC / Keck (K band, *under development*)

Radial Velocities
=================

Radial velocities are computed from reduced echelle spectra by forward modeling the individual orders (and optional cross-correlation).

Support Status:

1. iSHELL (*Kgas* mode, methane gas cell calibrated)
2. CHIRON (highres mode, R~136k, iodine gas cell)
3. Minerva-North (iodine gas cell calibrated)
4. Minerva-Australis (Pre-wavelength-calibrated via ThAr lamp, soon iodine gas cell)
5. NIRSPEC (K band, telluric calibrated, *under development*)
6. PARVI (Pre-wavelength-calibrated via LFC, *under development*)


Full Documentation -- https://pychell.readthedocs.io/en/latest/