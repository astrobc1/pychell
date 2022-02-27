# Base Python
import os
import glob

# Maths
import numpy as np

# Astropy
from astropy.io import fits
from astropy.time import Time
import astropy.units as units

# Pychell deps
import pychell.maths as pcmath

# Site
observatory = "cerro paranal"

# lsf sigma
lsf_sigma = [0.0005, 0.0013, 0.02]

######################
#### DATA PARSING ####
######################

def parse_header(input_file):
    return fits.open(input_file)[0].header

def parse_itime(data):
    return float(data.header["EXPTIME"])

def parse_object(data):
    return data.header["OBJECT"]
    
def parse_exposure_start_time(data):
    return Time(float(data.header['MJD-OBS']) + 2400000.5, scale='utc', format='jd').jd

def get_barycenter_corrections(data, star_name=None):
    bjd = float(data.header['HIERARCH ESO QC BJD'])
    bc_vel = float(data.header['HIERARCH ESO QC BERV']) * 1000
    return bjd, bc_vel

def estimate_wls(data, sregion):
    return data.wave

def parse_spec1d(input_file, sregion):
    f = fits.open(input_file)
    header = f[0].header
    wave = f[1].data.WAVE.flatten().astype(float) / 10
    flux = f[1].data.FLUX.flatten().astype(float)
    good = np.where((wave > sregion.wavemin) & (wave < sregion.wavemax))[0]
    wave = wave[good]
    wave = pcmath.doppler_shift_wave(wave, float(header['HIERARCH ESO DRS BERV']) * 1000)
    flux = flux[good]
    fluxerr = np.full(len(flux), 1E-3)
    mask = np.ones(len(flux))
    bad = np.where(~np.isfinite(flux) | ~np.isfinite(wave))[0]
    if bad.size > 0:
        mask[bad] = 0
    medval = pcmath.weighted_median(flux, percentile=0.99)
    flux /= medval
    data = {"wave": wave, "flux": flux, "fluxerr": fluxerr, "mask": mask}
    return data