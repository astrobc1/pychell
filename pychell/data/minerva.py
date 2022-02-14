# Base Python
import os
import glob

# Maths
import numpy as np

# Astropy
from astropy.io import fits
from astropy.coordinates import EarthLocation
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as units

# Barycorrpy
from barycorrpy import get_BC_vel
from barycorrpy.utc_tdb import JDUTC_to_BJDTDB

# Pychell deps
import pychell.maths as pcmath

#######################
#### Name and Site ####
#######################

observatory = {
    "name": "Whipple",
    "site": EarthLocation.of_site("Whipple")
}

echelle_orders = [94, 122]


######################
#### DATA PARSING ####
######################

def parse_image_num(data):
    string_list = data.base_input_file.split('.')
    data.image_num = string_list[4]
    return data.image_num
    
def parse_target(data):
    data.target = data.header["OBJECT"]
    return data.target
    
def parse_utdate(data):
    utdate = "".join(data.header["DATE_OBS"].split('-'))
    data.utdate = utdate
    return data.utdate
    
def parse_sky_coord(data):
    data.skycoord = SkyCoord(ra=data.header['TCS_RA'], dec=data.header['TCS_DEC'], unit=(units.hourangle, units.deg))
    return data.skycoord

def parse_exposure_start_time(data):
    data.time_obs_start = Time(float(data.header['JD']), scale='utc', format='jd')
    return data.time_obs_start

def parse_itime(data):
    data.itime = data.header["EXPTIME"]
    return data.itime
    
def parse_spec1d(data):
    fits_data = fits.open(data.input_file)[0]
    fits_data.verify('fix')
    data.header = fits_data.header
    oi = (echelle_orders[1] - echelle_orders[0]) - (data.order_num - echelle_orders[0])
    data.wave, data.flux, data.flux_unc, data.mask = fits_data.data[oi, :, 0].astype(np.float64), fits_data.data[oi, :, 1].astype(np.float64), fits_data.data[oi, :, 2].astype(np.float64), fits_data.data[oi, :, 3].astype(np.float64)

def parse_telescope(data):
    return int(data.base_input_file_noext[-1:])



################################
#### BARYCENTER CORRECTIONS ####
################################

def compute_barycenter_corrections(data, star_name):
        
    # Star name
    star_name = star_name.replace('_', ' ')
    
    # Compute the JD UTC mid point (possibly weighted)
    jdmid = compute_exposure_midpoint(data)
    
    # BJD
    bjd = JDUTC_to_BJDTDB(JDUTC=jdmid, starname=star_name, obsname=observatory['name'], leap_update=True)[0][0]
    
    # bc vel
    bc_vel = get_BC_vel(JDUTC=jdmid, starname=star_name, obsname=observatory['name'], leap_update=True)[0][0]
    
    # Add to data
    data.bjd = bjd
    data.bc_vel = bc_vel
    
    return bjd, bc_vel

def compute_exposure_midpoint(data):
    return parse_exposure_start_time(data).jd + parse_itime(data) / (2 * 86400)

#########################
#### BASIC WAVE INFO ####
#########################

def estimate_wls(data):
    wave = data.wave
    tel = parse_telescope(data)
    if tel == 4:
        return wave + 0.133
    else:
        return wave


#######################################
##### GENERATING RADIAL VELOCITIES ####
#######################################

rv_zero_point = -2410.0

lsf_width = [0.021, 0.0235, 0.026]