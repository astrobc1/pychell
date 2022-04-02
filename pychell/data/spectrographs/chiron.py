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
import pychell.spectralmodeling

# Site
observatory = "ctio"

# Orders
echelle_orders = [63, 125]

# Gas cell
gascell_file = "iodine_gas_cell_chiron_master_40K.npz"

# lsf sigma
lsf_sigma = [0.0009, 0.0014, 0.0018]

# Strange shifts between gas cell template and ThAr lamp (nm)
gas_cell_shifts = {63: -0.232527273, 64: -0.22998407699999998, 65: -0.227469068, 66: -0.22498224699999997, 67: -0.22252361399999998, 68: -0.220093168, 69: -0.21769090900000002, 70: -0.21531683799999998, 71: -0.212970954, 72: -0.210653258, 73: -0.20836374900000001, 74: -0.20610242799999998, 75: -0.20386929399999998, 76: -0.201664348, 77: -0.19948758900000002, 78: -0.197339018, 79: -0.195218634, 80: -0.193126437, 81: -0.191062428, 82: -0.189026606, 83: -0.187018972, 84: -0.185039526, 85: -0.183088267, 86: -0.181165195, 87: -0.179270311, 88: -0.177403614, 89: -0.175565104, 90: -0.173754783, 91: -0.171972648, 92: -0.170218701, 93: -0.168492942, 94: -0.16679537, 95: -0.165125985, 96: -0.163484788, 97: -0.161871779, 98: -0.160286957, 99: -0.15873032199999998, 100: -0.157201875, 101: -0.155701615, 102: -0.154229543, 103: -0.152785658, 104: -0.15136996, 105: -0.14998245100000002, 106: -0.148623128, 107: -0.147291993, 108: -0.145989046, 109: -0.144714286, 110: -0.143467713, 111: -0.142249328, 112: -0.14105913, 113: -0.13989712, 114: -0.138763298, 115: -0.13765766200000001, 116: -0.136580215, 117: -0.13553095399999998, 118: -0.134509881, 119: -0.133516996, 120: -0.13255229799999999, 121: -0.131615788, 122: -0.130707465, 123: -0.129827329, 124: -0.128975381, 125: -0.128151621}

######################
#### DATA PARSING ####
######################

def parse_header(input_file):
    return fits.open(input_file)[0].header

def parse_exposure_start_time(data):
    return Time(data.header['DATE-OBS']).jd

def parse_itime(data):
    return float(data.header["EXPTIME"])
    
def parse_spec1d(input_file, sregion):
    f = fits.open(input_file)[0]
    f.verify('fix')
    oi = (echelle_orders[1] - echelle_orders[0]) - (sregion.order - echelle_orders[0])
    wave = f.data[oi, :, 0].astype(float) / 10
    flux = f.data[oi, :, 1].astype(float)
    fluxerr = np.full(len(flux), 1E-3)
    mask = np.ones(len(flux))
    bad = np.where(~np.isfinite(flux) | ~np.isfinite(wave))[0]
    if bad.size > 0:
        mask[bad] = 0
    medval = pcmath.weighted_median(flux, percentile=0.99)
    flux /= medval
    data = {"wave": wave, "flux": flux, "fluxerr": fluxerr, "mask": mask}
    return data
    
def estimate_wls(data, sregion):
    wls = data.wave - gas_cell_shifts[sregion.order]
    return wls

def estimate_wls_from_file(input_file, order):
    oi = (echelle_orders[1] - echelle_orders[0]) - (order - echelle_orders[0])
    f = fits.open(input_file)[0]
    wave = f.data[oi, :, 0].astype(float) / 10
    wave -= gas_cell_shifts[order]
    return wave

def get_exposure_midpoint(data):
    return parse_exposure_start_time(data) + parse_itime(data) / (2 * 86400)

def get_barycenter_corrections(data, star_name):
    jdmid = get_exposure_midpoint(data)
    bjd, bc_vel = pychell.spectralmodeling.barycenter.compute_barycenter_corrections(jdmid, star_name, observatory)
    return bjd, bc_vel