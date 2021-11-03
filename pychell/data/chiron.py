# Base Python
import os
import glob

# Maths
import numpy as np

# Astropy
from astropy.io import fits
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

spectrograph = 'CHIRON'
observatory = {
    "name": 'CTIO',
    "lat": 30.169286111111113,
    "long": 70.806789,
    "alt": 2207
}

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
    oi = data.order_num - 1
    data.wave, data.flux = fits_data.data[oi, :, 0].astype(np.float64), fits_data.data[oi, :, 1].astype(np.float64)
    data.flux_unc = np.full(data.flux.shape, 1E-3)
    data.mask = np.ones_like(data.flux)
    
def estimate_wls(data):
    oi = data.order_num - 1
    shift = gas_cell_shifts[oi]
    wls = data.wave - shift
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
gas_cell_shifts = [-1.28151621, -1.28975381, -1.29827329, -1.30707465, -1.31615788, -1.32552298, -1.33516996, -1.34509881, -1.35530954, -1.36580215, -1.37657662, -1.38763298, -1.3989712, -1.4105913, -1.42249328, -1.43467713, -1.44714286, -1.45989046, -1.47291993, -1.48623128, -1.49982451, -1.5136996 , -1.52785658, -1.54229543, -1.55701615, -1.57201875, -1.58730322, -1.60286957, -1.61871779, -1.63484788, -1.65125985, -1.6679537 , -1.68492942, -1.70218701, -1.71972648, -1.73754783, -1.75565104, -1.77403614, -1.79270311, -1.81165195, -1.83088267, -1.85039526, -1.87018972, -1.89026606, -1.91062428, -1.93126437, -1.95218634, -1.97339018, -1.99487589, -2.01664348, -2.03869294, -2.06102428, -2.08363749, -2.10653258, -2.12970954, -2.15316838, -2.17690909, -2.20093168, -2.22523614, -2.24982247, -2.27469068, -2.29984077, -2.32527273]