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
    'name': 'IRTF',
    'lat': 19.826218316666665,
    'lon': -155.4719987888889,
    'alt': 4168.066848
}


######################
#### DATA PARSING ####
######################

def categorize_raw_data(data_input_path, output_path):

    # Stores the data as above objects
    data = {}
    
    # iSHELL science files are files that contain spc or data
    sci_files = glob.glob(data_input_path + "*data*.fits") + glob.glob(data_input_path + "*spc*.fits")
    sci_files = np.sort(np.unique(np.array(sci_files, dtype='<U200'))).tolist()
    data['science'] = [pcspecdata.RawEchellogram(input_file=sci_file, specmod=sys.modules[__name__]) for sci_file in sci_files]

    # Darks assumed to contain dark in filename
    dark_files = glob.glob(data_input_path + '*dark*.fits')
    if len(dark_files) > 0:
        data['darks'] = [pcspecdata.RawEchellogram(input_file=dark_files[f], specmod=sys.modules[__name__]) for f in range(len(dark_files))]
        dark_groups = group_darks(data['darks'])
        data['master_darks'] = [pcspecdata.MasterCal(dark_group, output_path + "calib" + os.sep) for dark_groups in dark_group]
    
        for sci in data['science']:
            pair_master_dark(sci, data['master_darks'])
        
        for flat in data['flats']:
            pair_master_dark(flat, data['master_darks'])
    
    # iSHELL flats must contain flat in the filename
    flat_files = glob.glob(data_input_path + '*flat*.fits')
    if len(flat_files) > 0:
        data['flats'] = [pcspecdata.RawEchellogram(input_file=flat_files[f], specmod=sys.modules[__name__]) for f in range(len(flat_files))]
        flat_groups = group_flats(data['flats'])
        data['master_flats'] = [pcspecdata.MasterCal(flat_group, output_path + "calib" + os.sep) for flat_group in flat_groups]
    
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
    
def pair_order_maps(data, order_maps):
    for order_map in order_maps:
        if order_map == data.master_flat:
            data.order_maps = [order_map]

def parse_image_num(data):
    string_list = data.base_input_file.split('.')
    data.image_num = string_list[4]
    return data.image_num
    
def parse_itime(data):
    data.itime = data.header["ITIME"]
    return data.itime

def parse_object(data):
    data.object = data.header["OBJECT"]
    return data.object
    
def parse_utdate(data):
    utdate = "".join(data.header["DATE_OBS"].split('-'))
    data.utdate = utdate
    return data.utdate

def parse_sky_coord(data):
    data.skycoord = SkyCoord(ra=data.header['TCS_RA'], dec=data.header['TCS_DEC'], unit=(units.hourangle, units.deg))
    return data.skycoord
    
def parse_exposure_start_time(data):
    data.time_obs_start = Time(float(data.header['TCS_UTC']) + 2400000.5, scale='utc', format='jd')
    return data.time_obs_start

def gen_master_calib_filename(master_cal):
    fname0 = master_cal.group[0].base_input_file.lower()
    if "dark" in fname0:
        return f"master_dark_{master_cal.group[0].utdate}{group[0].itime}s.fits"
    elif "flat" in fname0:
        img_nums = np.array([parse_image_num(d) for d in master_cal.group], dtype=int)
        img_start, img_end = img_nums.min(), img_nums.max()
        return f"master_flat_{master_cal.group[0].utdate}_imgs{img_start}-{img_end}.fits"
    else:
        return f"master_calib_{master_cal.group[0].utdate}.fits"

def gen_master_calib_header(master_cal):
    master_cal.skycoord = master_cal.group[0].skycoord
    master_cal.time_obs_start = master_cal.group[0].time_obs_start
    master_cal.object = master_cal.group[0].object
    master_cal.itime = master_cal.group[0].itime
    return copy.deepcopy(master_cal.group[0].header)

def pair_master_dark(data, master_darks):
    itimes = np.array([master_darks[i].itime for i in range(len(master_darks))], dtype=float)
    good = np.where(data.itime == itimes)[0]
    if good.size != 1:
        raise ValueError(str(good.size) + " master dark(s) found for\n" + str(data))
    else:
        data.master_dark = master_darks[good[0]]
    
def pair_master_flat(data, master_flats):
    ang_seps = np.array([np.abs(data.skycoord.separation(master_flat.skycoord)).value for master_flat in master_flats], dtype=float)
    ang_seps /= np.nanmax(ang_seps)
    time_seps = np.array([np.abs(data.time_obs_start.value - master_flat.time_obs_start.value) for master_flat in master_flats], dtype=float)
    time_seps /= np.nanmax(time_seps)
    ds = np.sqrt(ang_seps**2 + time_seps**2)
    minds_loc = np.argmin(ds)
    data.master_flat = master_flats[minds_loc]
  
def group_darks(darks):
    groups = []
    itimes = np.array([dark.itime for dark in darks])
    itimes_unq = np.unique(itimes)
    for t in itimes_unq:
        good = np.where(itimes == t)[0]
        indiv_darks = [darks[i] for i in good]
        groups.append(indiv_darks)
    return groups
    
def group_flats(flats):
    
    # Groups
    groups = []
    
    # Number of total flats
    n_flats = len(flats)
    
    # Create a clustering object
    density_cluster = sklearn.cluster.DBSCAN(eps=0.01745, min_samples=2, metric='euclidean', algorithm='auto', p=None, n_jobs=1)
    
    # Points are the ra and dec and time
    dist_matrix = np.empty(shape=(n_flats, n_flats), dtype=float)
    for i in range(n_flats):
        for j in range(n_flats):
            dpsi = np.abs(flats[i].skycoord.separation(flats[j].skycoord).value)
            dt = np.abs(flats[i].time_obs_start.jd - flats[j].time_obs_start.jd)
            dpsi /= np.pi
            dt /= 10  # Places more emphasis on delta psi
            dist_matrix[i, j] = np.sqrt(dpsi**2 + dt**2)
    
    # Fit
    db = density_cluster.fit(dist_matrix)
    
    # Extract the labels
    labels = db.labels_
    good_labels = np.where(labels >= 0)[0]
    if good_labels.size == 0:
        raise ValueError('The flat pairing algorithm failed!')
    good_labels_init = labels[good_labels]
    labels_unique = np.unique(good_labels_init)
    
    # The number of master flats
    n_mflats = len(labels_unique)

    for l in range(n_mflats):
        this_label = np.where(good_labels_init == labels_unique[l])[0]
        indiv_flats = [flats[lb] for lb in this_label]
        groups.append(indiv_flats)
        
    return groups
    
def pair_master_bias(data, master_bias):
    data.master_bias = master_bias

def parse_spec1d(data):
    
    # Load the flux, flux unc, and bad pix arrays
    fits_data = fits.open(data.input_file)[0]
    fits_data.verify('fix')
    data.header = fits_data.header
    oi = data.order_num - 1
    data.flux, data.flux_unc, data.mask = fits_data.data[oi, 0, :, 0].astype(np.float64), fits_data.data[oi, 0, :, 1].astype(np.float64), fits_data.data[oi, 0, :, 2].astype(np.float64)

    # Flip the data so wavelength is increasing for iSHELL data
    data.flux = data.flux[::-1]
    data.mask = data.mask[::-1]
    data.flux_unc = data.flux_unc[::-1]

def parse_image(data):
    image = fits.open(data.input_file, do_not_scale_image_data=True)[0].data.astype(float)
    correct_readmath(data, image)
    return image

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

def parse_fiber_nums(data):
    return None

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
    oi = data.order_num - 1
    waves = np.array([quad_set_point_1[oi], quad_set_point_2[oi], quad_set_point_3[oi]])
    pcoeffs = pcmath.poly_coeffs(quad_pixel_set_points, waves)
    wls = np.polyval(pcoeffs, np.arange(data.flux.size))
    return wls


################################
#### REDUCTION / EXTRACTION ####
################################

# List of detectors.
read_noise = 8.0
dark_current = 0.05
gain = 1.8


#######################################
##### GENERATING RADIAL VELOCITIES ####
#######################################

# Gas dell depth
gas_cell_depth = [0.97, 0.97, 0.97]

# Gas cell file
gas_cell_file = "methane_gas_cell_ishell_kgas.npz"

# LSF width
lsf_width = [0.08, 0.11, 0.15]

# RV Zero point (approx)
rv_zero_point = -6817.0


# Information to generate a crude ishell wavelength solution for the above method estimate_wavelength_solution
quad_pixel_set_points = [199, 1023.5, 1847]

# Left most set point for the quadratic wavelength solution
quad_set_point_1 = [24545.57561435, 24431.48444449, 24318.40830764, 24206.35776048, 24095.33986576, 23985.37381209, 23876.43046386, 23768.48974584, 23661.54443537, 23555.56359209, 23450.55136357, 23346.4923953, 23243.38904298, 23141.19183839, 23039.90272625, 22939.50127095, 22840.00907242, 22741.40344225, 22643.6481698, 22546.74892171, 22450.70934177, 22355.49187891, 22261.08953053, 22167.42305394, 22074.72848136, 21982.75611957, 21891.49178289, 21801.07332421, 21711.43496504]

# Middle set point for the quadratic wavelength solution
quad_set_point_2 = [24628.37672608, 24513.79686837, 24400.32734124, 24287.85495107, 24176.4424356, 24066.07880622, 23956.7243081, 23848.39610577, 23741.05658955, 23634.68688897, 23529.29771645, 23424.86836784, 23321.379387, 23218.80573474, 23117.1876433, 23016.4487031, 22916.61245655, 22817.65768889, 22719.56466802, 22622.34315996, 22525.96723597, 22430.41612825, 22335.71472399, 22241.83394135, 22148.73680381, 22056.42903627, 21964.91093944, 21874.20764171, 21784.20091295]

# Right most set point for the quadratic wavelength solution
quad_set_point_3 = [24705.72472863, 24590.91231465, 24476.99298677, 24364.12010878, 24252.31443701, 24141.55527091, 24031.82506843, 23923.12291214, 23815.40789995, 23708.70106907, 23602.95596074, 23498.18607941, 23394.35163611, 23291.44815827, 23189.49231662, 23088.42080084, 22988.26540094, 22888.97654584, 22790.57559244, 22693.02942496, 22596.33915038, 22500.49456757, 22405.49547495, 22311.25574559, 22217.91297633, 22125.33774808, 22033.50356525, 21942.41058186, 21852.24253555]