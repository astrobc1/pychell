# Base Python
import os
import importlib
import sys
import copy
import glob

# Astropy
from astropy.io import fits
from astropy.coordinates import EarthLocation
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as units

# Maths
import numpy as np
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
    'site': EarthLocation.of_site("Cerro Paranal")
}

echelle_orders = None

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


################################
#### BARYCENTER CORRECTIONS ####
################################

def compute_barycenter_corrections(data, star_name):
        
    # Star name
    star_name = star_name.replace('_', ' ')
    
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