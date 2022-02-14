# Base Python
import os
import importlib
import copy
import glob
import sys

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
import pychell.data.spectraldata as pcspecdata

#############################
####### Name and Site #######
#############################

spectrograph = 'NIRSPEC'
observatory = {
    "name": "Keck"
}

utc_offset = -10
    
def categorize_raw_data(data_input_path, output_path):

    # Stores the data as above objects
    data = {}
    
    # iSHELL science files are files that contain spc or data
    sci_files = glob.glob(data_input_path + "*data*.fits")
    data['science'] = [pcspecdata.RawEchellogram(input_file=sci_file, specmod=sys.modules[__name__]) for sci_file in sci_files]
        
    # Darks assumed to contain dark in filename
    dark_files = glob.glob(data_input_path + '*dark*.fits')

    # NIRSPEC flats must contain flat in the filename
    flat_files = glob.glob(data_input_path + '*flat*.fits')


    if len(dark_files) > 0:
        data['darks'] = [pcspecdata.RawEchellogram(input_file=dark_files[f], specmod=sys.modules[__name__]) for f in range(len(dark_files))]
        dark_groups = group_darks(data['darks'])
        data['master_darks'] = [pcspecdata.MasterCal(dark_group, output_path + "calib" + os.sep) for dark_groups in dark_group]

    if len(flat_files) > 0:
        data['flats'] = [pcspecdata.RawEchellogram(input_file=flat_files[f], specmod=sys.modules[__name__]) for f in range(len(flat_files))]
        flat_groups = group_flats(data['flats'])
        data['master_flats'] = [pcspecdata.MasterCal(flat_group, output_path + "calib" + os.sep) for flat_group in flat_groups]


    if len(dark_files) > 0:
        for sci in data['science']:
            pair_master_dark(sci, data['master_darks'])
        

        if len(flat_files) > 0:
            for flat in data['flats']:
                pair_master_dark(flat, data['master_darks'])

    if len(flat_files) > 0:
        for sci in data['science']:
            pair_master_flat(sci, data['master_flats'])
        
    # Order maps for iSHELL are the flat fields closest in time and space (RA+Dec) to the science target
    data['order_maps'] = data['master_flats']
    for sci_data in data['science']:
        pair_order_maps(sci_data, data['order_maps'])

    # Which to extract
    data['extract'] = data['science']
    
    # Print reduction summary
    print_reduction_summary(data)

    # Return the data dict
    return data

def group_flats(flats):
    return [flats]

def pair_order_maps(data, order_maps):
    for order_map in order_maps:
        if order_map == data.master_flat:
            data.order_maps = [order_map]

def parse_image_num(data):
    string_list = data.base_input_file.split('.')
    data.image_num = string_list[1]
    return data.image_num
    
def parse_itime(data):
    data.itime = data.header["ITIME"]
    return data.itime

def pair_master_flat(data, master_flats):
    data.master_flat = master_flats[0]

def parse_object(data):
    data.object = data.header["OBJECT"].replace(" ", "")
    return data.object
    
def parse_utdate(data):
    utdate = "".join(data.header["DATE-OBS"].split('-'))
    data.utdate = utdate
    return data.utdate
    
def parse_sky_coord(data):
    data.skycoord = SkyCoord(ra=data.header['RA'], dec=data.header['DEC'], unit=(units.hourangle, units.deg))
    return data.skycoord
    
def parse_exposure_start_time(data):
    data.time_obs_start = Time(float(data.header['MJD-OBS']) + 2400000.5, scale='utc', format='jd')
    return data.time_obs_start

def gen_master_calib_filename(master_cal):
    fname0 = master_cal.group[0].base_input_file.lower()
    if "dark" in fname0:
        return f"master_dark_{master_cal.group[0].utdate}{group[0].itime}s.fits"
    elif "flat" in fname0:
        img_nums = np.array([parse_image_num(d) for d in master_cal.group], dtype=int)
        img_start, img_end = img_nums.min(), img_nums.max()
        return f"master_flat_{master_cal.group[0].utdate}imgs{img_start}-{img_end}.fits"
    else:
        return f"master_calib_{master_cal.group[0].utdate}.fits"

def gen_master_calib_header(master_cal):
    master_cal.skycoord = master_cal.group[0].skycoord
    master_cal.time_obs_start = master_cal.group[0].time_obs_start
    master_cal.object = master_cal.group[0].object
    master_cal.itime = master_cal.group[0].itime
    return copy.deepcopy(master_cal.group[0].header)
    
def parse_spec1d(data):
    
    # Load the flux, flux unc, and bad pix arrays
    fits_data = fits.open(data.input_file, output_verify='ignore')[0]
    data.header = fits_data.header
    oi = data.order_num - 1
    data.flux, data.flux_unc, data.mask = fits_data.data[oi, 0, :, 0].astype(np.float64), fits_data.data[oi, 0, :, 1].astype(np.float64), fits_data.data[oi, 0, :, 2].astype(np.float64)

def parse_image(data):
    image = fits.open(data.input_file, do_not_scale_image_data=True)[0].data.astype(float)
    correct_readmath(data, image)
    return image

def correct_readmath(data, data_image):
    # Corrects NDRs - Number of dynamic reads, or Non-destructive reads, take your pick.
    # This reduces the read noise by sqrt(NDR)
    if hasattr(data, "header"):
        if 'NDR' in data.header:
            data_image /= float(data.header['NDR'])
        if 'BZERO' in data.header:
            data_image -= float(data.header['BZERO'])
        if 'BSCALE' in data.header:
            data_image /= float(data.header['BSCALE'])

def parse_image_header(data):
        
    # Parse the fits HDU
    fits_hdu = fits.open(data.input_file)[0]
    
    # Just in case
    try:
        fits_hdu.verify('fix')
    except:
        pass
    
    # Store the header
    data.header = fits_hdu.header
    
    # Parse the sky coord and time of obs
    parse_utdate(data)
    parse_sky_coord(data)
    parse_exposure_start_time(data)
    parse_object(data)
    parse_itime(data)
    
    return data.header

def print_reduction_summary(data):
    
    n_sci_tot = len(data['science'])
    targets_all = np.array([data['science'][i].object for i in range(n_sci_tot)], dtype='<U50')
    targets_unique = np.unique(targets_all)
    for i in range(len(targets_unique)):
        
        target = targets_unique[i]
        
        locs_this_target = np.where(targets_all == target)[0]
        
        sci_this_target = [data['science'][j] for j in locs_this_target]
        
        print('Target: ' + target)
        print('    N Exposures: ' + str(locs_this_target.size))
        if hasattr(sci_this_target[0], 'master_bias'):
            print('  Master Bias: ')
            print('    ' + str(sci_this_target[0].master_bias))
            
        if hasattr(sci_this_target[0], 'master_dark'):
            print('  Master Dark: ')
            print('    ' + str(sci_this_target[0].master_dark))
            
        if hasattr(sci_this_target[0], 'master_flat'):
            print('  Master Flat: ')
            print('    ' + str(sci_this_target[0].master_flat))


def compute_exposure_midpoint(data):
    return parse_exposure_start_time(data).jd + parse_itime(data) / (2 * 86400)

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

#########################
#### BASIC WAVE INFO ####
#########################

def estimate_wls(data):
    oi = data.order_num - 1
    waves = np.array([quad_set_point_1[oi], quad_set_point_2[oi], quad_set_point_3[oi]])
    pcoeffs = pcmath.poly_coeffs(quad_pixel_set_points, waves)
    wls = np.polyval(pcoeffs, np.arange(data.flux.size))
    return wls

read_noise = 0

###########################
#### RADIAL VELOCITIES ####
###########################

lsf_width = [0.1, 0.25, 0.4]
rv_zero_point = 0

# Information to generate a crude ishell wavelength solution for the above method estimate_wavelength_solution
quad_pixel_set_points = [1, 512, 1023]

# Left most set point for the quadratic wavelength solution
quad_set_point_1 = np.array([19900.00 - 36, 19900.00 - 36 + 470, 20600.00,21500.00,22200.00,22800.00,23600.00])

# Middle set point for the quadratic wavelength solution
quad_set_point_2 = np.array([20050.00 - 40.5, 20050.00 - 40.5 + 470, 20950.00, 21600.00, 22350.00, 23000.00, 23750.00])

# Right most set point for the quadratic wavelength solution
quad_set_point_3 = np.array([20200.00 - 40, 20200.00 - 40 + 470, 21300.00, 21700.00, 22500.00, 23200.00, 23900.00])