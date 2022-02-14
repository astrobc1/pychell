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
#### NAME AND SITE ####
#######################

observatory = {
    "name": "ctio",
    "site" : EarthLocation.of_site("ctio"),
}

echelle_orders = [63, 125]

######################
#### DATA PARSING ####
######################

def parse_exposure_start_time(data):
    data.time_obs_start = Time(data.header['DATE-OBS'])
    return data.time_obs_start

def parse_itime(data):
    data.itime = data.header["EXPTIME"]
    return data.itime
    
def parse_spec1d(data):
    fits_data = fits.open(data.input_file)[0]
    fits_data.verify('fix')
    data.header = fits_data.header
    oi = (echelle_orders[1] - echelle_orders[0]) - (data.order_num - echelle_orders[0])
    data.wave, data.flux = fits_data.data[oi, :, 0].astype(np.float64), fits_data.data[oi, :, 1].astype(np.float64)
    data.flux_unc = np.full(data.flux.shape, 1E-3)
    data.mask = np.ones_like(data.flux)
    
def estimate_wls(data):
    wls = data.wave - gas_cell_shifts[data.order_num]
    return wls


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


#######################################
##### GENERATING RADIAL VELOCITIES ####
#######################################

lsf_width = [0.009, 0.014, 0.018]
rv_zero_point = -8_000.0

###############
#### MISC. ####
###############

# Shifts between the thar lamp and gas cell for each order
#gas_cell_shifts = [-1.28151621, -1.28975381, -1.29827329, -1.30707465, -1.31615788, -1.32552298, -1.33516996, -1.34509881, -1.35530954, -1.36580215, -1.37657662, -1.38763298, -1.3989712, -1.4105913, -1.42249328, -1.43467713, -1.44714286, -1.45989046, -1.47291993, -1.48623128, -1.49982451, -1.5136996 , -1.52785658, -1.54229543, -1.55701615, -1.57201875, -1.58730322, -1.60286957, -1.61871779, -1.63484788, -1.65125985, -1.6679537 , -1.68492942, -1.70218701, -1.71972648, -1.73754783, -1.75565104, -1.77403614, -1.79270311, -1.81165195, -1.83088267, -1.85039526, -1.87018972, -1.89026606, -1.91062428, -1.93126437, -1.95218634, -1.97339018, -1.99487589, -2.01664348, -2.03869294, -2.06102428, -2.08363749, -2.10653258, -2.12970954, -2.15316838, -2.17690909, -2.20093168, -2.22523614, -2.24982247, -2.27469068, -2.29984077, -2.32527273]

gas_cell_shifts = {63: -2.32527273, 64: -2.29984077, 65: -2.27469068, 66: -2.24982247, 67: -2.22523614, 68: -2.20093168, 69: -2.17690909, 70: -2.15316838, 71: -2.12970954, 72: -2.10653258, 73: -2.08363749, 74: -2.06102428, 75: -2.03869294, 76: -2.01664348, 77: -1.99487589, 78: -1.97339018, 79: -1.95218634, 80: -1.93126437, 81: -1.91062428, 82: -1.89026606, 83: -1.87018972, 84: -1.85039526, 85: -1.83088267, 86: -1.81165195, 87: -1.79270311, 88: -1.77403614, 89: -1.75565104, 90: -1.73754783, 91: -1.71972648, 92: -1.70218701, 93: -1.68492942, 94: -1.6679537, 95: -1.65125985, 96: -1.63484788, 97: -1.61871779, 98: -1.60286957, 99: -1.58730322, 100: -1.57201875, 101: -1.55701615, 102: -1.54229543, 103: -1.52785658, 104: -1.5136996, 105: -1.49982451, 106: -1.48623128, 107: -1.47291993, 108: -1.45989046, 109: -1.44714286, 110: -1.43467713, 111: -1.42249328, 112: -1.4105913, 113: -1.3989712, 114: -1.38763298, 115: -1.37657662, 116: -1.36580215, 117: -1.35530954, 118: -1.34509881, 119: -1.33516996, 120: -1.32552298, 121: -1.31615788, 122: -1.30707465, 123: -1.29827329, 124: -1.28975381, 125: -1.28151621}