# Base Python
import os
import importlib
import sys
import copy
import glob

# Astropy fits object
from astropy.io import fits

# Maths
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as units
import sklearn.cluster

# Barycorrpy
from barycorrpy import get_BC_vel
from barycorrpy.utc_tdb import JDUTC_to_BJDTDB

# Pychell deps
import pychell.data.spectraldata as pcspecdata
import pychell.maths as pcmath


##############
#### SITE ####
##############

observatory = {
    'name': 'Cerro Paranal',
    'lat': None,
    'lon': None,
    'alt': None
}

utc_offset = -3

######################
#### DATA PARSING ####
######################

def parse_itime(data):
    data.itime = data.header["EXPTIME"]
    return data.itime

def parse_object(data):
    data.object = data.header["OBJECT"]
    return data.object
    
def parse_exposure_start_time(data):
    data.time_obs_start = Time(float(data.header['MJD-OBS']) + 2400000.5, scale='utc', format='jd')
    return data.time_obs_start

# def parse_spec1d(data):
    
#     # Load the flux, flux unc, and bad pix arrays
#     fits_data = fits.open(data.input_file)
#     fits_data.verify('fix')
#     data.header = fits_data[0].header
#     oi = data.order_num - 1
#     #data.wave = fits_data[1].data.WAVE.flatten().astype(float)
#     #data.flux = fits_data[1].data.FLUX_EL_SKYSUB.flatten().astype(float)
#     #data.flux_unc = fits_data[1].data.ERR.flatten().astype(float)
#     #data.wave = fits_data[1].data.WAVE.flatten().astype(float)
#     #data.flux = fits_data[1].data.FLUX_EL_SKYSUB.flatten().astype(float)


#     data.flux_unc = np.full(len(data.wave), 1E-3)
#     data.mask = np.ones(len(data.wave))
#     good = np.where(data.wave > 4500)[0]
#     data.wave = data.wave[good]
#     data.flux = data.flux[good]
#     data.flux_unc = data.flux_unc[good]
#     data.mask = data.mask[good]

#     data.wave = pcmath.doppler_shift_wave(data.wave, -1 * data.header['HIERARCH ESO QC BERV'])

#ECHELLE_ORDERS = [212, 240]


################################
#### BARYCENTER CORRECTIONS ####
################################

def compute_barycenter_corrections(data, star_name):
        
    # Star name
    star_name = star_name.replace('_', ' ')
    
    # Compute the JD UTC mid point (possibly weighted)
    #jdmid = compute_exposure_midpoint(data)
    
    # BJD
    #bjd = JDUTC_to_BJDTDB(JDUTC=jdmid, starname=star_name, obsname=observatory['name'], leap_update=True)[0][0]
    
    # bc vel
    #bc_vel = get_BC_vel(JDUTC=jdmid, starname=star_name, obsname=observatory['name'], leap_update=True)[0][0]
    
    # Add to data
    bjd = data.header['HIERARCH ESO QC BJD']
    bc_vel = data.header['HIERARCH ESO QC BERV'] * 1000
    data.bjd = bjd
    data.bc_vel = bc_vel
    
    return bjd, bc_vel

def compute_exposure_midpoint(data):
    return parse_exposure_start_time(data).jd + parse_itime(data) / (2 * 86400)


#########################
#### BASIC WAVE INFO ####
#########################

def estimate_wls(data):
    return data.wave


################################
#### REDUCTION / EXTRACTION ####
################################


#######################################
##### GENERATING RADIAL VELOCITIES ####
#######################################

# LSF width
lsf_width = [0.008, 0.013, 0.2]

# RV Zero point for stellar template (don't yet know why this is needed - what are simbad absolute rvs relative to?)
rv_zero_point = 0