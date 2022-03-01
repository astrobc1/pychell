# Base Python
import os
import glob

import pychell.spectralmodeling.barycenter

# Maths
import numpy as np

# Astropy
from astropy.io import fits
from astropy.time import Time
import astropy.units as units

# Pychell deps
import pychell.maths as pcmath

# Site
observatory = "whipple"

# Orders
echelle_orders = [94, 122]

# Gas cell
gascell_file = "iodine_gas_cell_minervanorth_nist.npz"

# lsf sigma
lsf_sigma = [0.0015, 0.0021, 0.0025]


######################
#### DATA PARSING ####
######################

def parse_header(input_file):
    return fits.open(input_file)[0].header

def parse_exposure_start_time(data):
    return Time(float(data.header['JD']), scale='utc', format='jd').jd

def parse_itime(data):
    return float(data.header["EXPTIME"])

def parse_spec1d(input_file, sregion):
    f = fits.open(input_file)[0]
    f.verify('fix')
    oi = (echelle_orders[1] - echelle_orders[0]) - (sregion.order - echelle_orders[0])
    wave = f.data[oi, :, 0].astype(float) / 10
    flux = f.data[oi, :, 1].astype(float)
    fluxerr = f.data[oi, :, 2].astype(float)
    mask = f.data[oi, :, 3].astype(float) 
    bad = np.where(~np.isfinite(flux) | ~np.isfinite(wave))[0]
    if bad.size > 0:
        mask[bad] = 0
    medval = pcmath.weighted_median(flux, percentile=0.99)
    flux /= medval
    fluxerr /= medval
    data = {"wave": wave, "flux": flux, "fluxerr": fluxerr, "mask": mask}
    return data

def parse_telescope(data):
    return int(data.base_input_file_noext[-1:])

def get_barycenter_corrections(data, star_name):
    jdmid = get_exposure_midpoint(data)
    bjd, bc_vel = pychell.spectralmodeling.barycenter.compute_barycenter_corrections(jdmid, star_name, observatory)
    return bjd, bc_vel

def get_exposure_midpoint(data):
    return parse_exposure_start_time(data) + parse_itime(data) / (2 * 86400)

def estimate_wls(data, sregion):
    wls = data.wave
    tel = parse_telescope(data)
    if tel == 4:
        return wls + 0.0133
    return wls

def estimate_wls_from_file(input_file, order):
    oi = (echelle_orders[1] - echelle_orders[0]) - (order - echelle_orders[0])
    f = fits.open(input_file)[0]
    wave = f.data[oi, :, 0].astype(float) / 10
    return wave