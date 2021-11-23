# Base Python
import os
import copy
import glob
import sys

# Maths
import numpy as np
import scipy.constants as cs

# Astropy
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as units
from astropy.io import fits

# Barycorrpy
from barycorrpy import get_BC_vel
from barycorrpy.utc_tdb import JDUTC_to_BJDTDB

# Pychell deps
import pychell.maths as pcmath
import pychell.data.spectraldata as pcspecdata


#######################
#### NAME AND SITE ####
#######################

spectrograph = "PARVI"
observatory = {
    "name" : "Palomar",
    "lat": 33.3537819182,
    "long": -116.858929898,
    "alt": 1713.0
}

utc_offset = -8

######################
#### DATA PARSING ####
######################

def categorize_raw_data(data_input_path, output_path):

    # Stores the data as above objects
    data = {}

    # Classify files
    all_files = glob.glob(data_input_path + '*.fits')
    lfc_files = glob.glob(data_input_path + '*LFC_*.fits') + glob.glob(data_input_path + '*LFCSTABILITY_*.fits')
    dark_files = glob.glob(data_input_path + '*DARK*.fits')
    full_flat_files = glob.glob(data_input_path + '*FULLFLAT*.fits')
    badpix_files = glob.glob(data_input_path + "*BadPixels*.fits")
    fiber_flat_files = glob.glob(data_input_path + '*FIBERFLAT*.fits') + glob.glob(data_input_path + '*FIBREFLAT*.fits')
    sci_files = list(set(all_files) - set(lfc_files) - set(dark_files) - set(full_flat_files) - set(badpix_files) - set(fiber_flat_files))

    # Create Echellograms from raw data
    data['science'] = [pcspecdata.RawEchellogram(input_file=f, spectrograph=spectrograph) for f in sci_files]
    data['fiber_flats'] = [pcspecdata.RawEchellogram(input_file=f, spectrograph=spectrograph) for f in fiber_flat_files]
    data['darks'] = [pcspecdata.RawEchellogram(input_file=f, spectrograph=spectrograph) for f in dark_files]
    data['flats'] = [pcspecdata.RawEchellogram(input_file=f, spectrograph=spectrograph) for f in full_flat_files]
    data['lfc'] = [pcspecdata.RawEchellogram(input_file=f, spectrograph=spectrograph) for f in lfc_files]
    
    # Master Darks
    if len(dark_files) > 0:
        data['master_darks'] = [pcspecdata.MasterCal(group, output_path + "calib" + os.sep) for group in group_darks(data['darks'])]

    # Master Flats
    if len(full_flat_files) > 0:
        data['master_flats'] = [pcspecdata.MasterCal(group, output_path + "calib" + os.sep) for group in group_flats(data['flats'])]

    # Order maps
    data['order_maps'] = data['fiber_flats']

    # Which to extract
    data['extract'] = data['science'] + data['fiber_flats'] + data['lfc']

    # Pair order maps for the spectra to extract
    for d in data['extract']:
        pair_order_maps(d, data['extract'])

    # Pair darks with full frame flats
    if len(full_flat_files) > 0 and len(dark_files) > 0:
        for flat in data['flats']:
            pair_master_dark(flat, data['master_darks'])

    # Pair darks and flats with all extract (sci, LFC, fiber flats)
    for sci in data['extract']:
        if len(dark_files) > 0:
            pair_master_dark(sci, data['master_darks'])
        if len(full_flat_files) > 0:
            pair_master_flat(sci, data['master_flats'])

    # Bad pixel mask (only one, load into memory)
    data['badpix_mask'] = 1 - fits.open(badpix_files[0])[0].data.astype(float)
        

    #self.print_summary(data)

    return data

def group_darks(darks):
    return [darks]

def group_flats(flats):
    return [flats]

def gen_master_calib_filename(master_cal):
    fname0 = master_cal.group[0].base_input_file.lower()
    if "dark" in fname0:
        return f"master_dark_{master_cal.group[0].utdate}{master_cal.group[0].itime}s.fits"
    elif "fiberflat" in fname0 or "fibreflat" in fname0:
        return f"master_fiberflat_{master_cal.group[0].utdate}.fits"
    elif "fullflat" in fname0:
        return f"master_fullflat_{master_cal.group[0].utdate}.fits"
    elif "lfc" in fname0:
        return f"master_lfc_{master_cal.group[0].utdate}.fits"
    else:
        return f"master_calib_{master_cal.group[0].utdate}.fits"

def gen_master_calib_header(master_cal):
    return copy.deepcopy(master_cal.group[0].header)

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

def parse_image(data):
    image = fits.open(data.input_file, do_not_scale_image_data=True)[0].data.astype(float).T
    return image

def pair_master_dark(data, master_darks):
    data.master_dark = master_darks[0]

def pair_master_flat(data, master_flats):
    data.master_flat = master_flats[0]

def pair_order_maps(data, order_maps):
    fibers_sci = [int(f) for f in str(parse_fiber_nums(data))]
    fibers_order_maps = [int(parse_fiber_nums(order_map)) for order_map in order_maps]
    n_fibers_sci = len(fibers_sci)
    order_maps_out = []
    for fiber in fibers_sci:
        k = fibers_order_maps.index(fiber)
        if k == -1:
            raise ValueError(f"No fiber flat corresponding to {data}")
        else:
            order_maps_out.append(order_maps[k])
    data.order_maps = order_maps_out

def parse_image_num(data):
    return 1
    
def parse_object(data):
    data.object = data.header["OBJECT"]
    return data.object
    
def parse_utdate(data):
    utdate = "".join(data.header["P200_UTC"].split('T')[0].split("-"))
    data.utdate = utdate
    return data.utdate
    
def parse_sky_coord(data):
    if data.header['P200RA'] is not None and data.header['P200DEC'] is not None:
        data.skycoord = SkyCoord(ra=data.header['P200RA'], dec=data.header['P200DEC'], unit=(units.hourangle, units.deg))
    else:
        data.skycoord = SkyCoord(ra=np.nan, dec=np.nan, unit=(units.hourangle, units.deg))
    return data.skycoord

def parse_itime(data):
    data.itime = data.header["EXPTIME"]
    return data.itime
    
def parse_exposure_start_time(data):
    data.time_obs_start = Time(float(data.header["TIMEI00"]) / 1E9, format="unix")
    return data.time_obs_start

def parse_fiber_nums(data):
    return int(data.header["FIBER"])
    
def parse_spec1d(data):
    fits_data = fits.open(data.input_file)
    fits_data.verify('fix')
    data.header = fits_data[0].header
    oi = data.order_num - 1
    data.wave = fits_data[0].data[oi, :, 0]
    data.flux = fits_data[0].data[oi, :, 1] / fits_data[0].data[oi, :, 4]
    data.flux_unc = fits_data[0].data[oi, :, 2]
    data.mask = fits_data[0].data[oi, :, 3]
    #data.lsf_width = fits_data[1].data[oi]

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
    jdsi, jdsf = [], []
    for key in data.header:
        if key.startswith("TIMEI"):
            jdsi.append(Time(float(data.header[key]) / 1E9, format="unix").jd)
        if key.startswith("TIMEF"):
            jdsf.append(Time(float(data.header[key]) / 1E9, format="unix").jd)
    mean_jd = (np.nanmax(jdsf) - np.nanmin(jdsi)) / 2 + np.nanmin(jdsi)
    return mean_jd

###################
#### WAVE INFO ####
###################


def estimate_wls(data=None, order_num=None, fiber_num=1):
    if order_num is None:
        order_num = data.order_num
    if fiber_num == 1:
        pcoeffs = wls_coeffs_fiber1[order_num - 1 + 24]
    else:
        pcoeffs = wls_coeffs_fiber3[order_num - 1 + 24]
    wls = np.polyval(pcoeffs, np.arange(2048).astype(float))
    return wls


################################
#### REDUCTION / EXTRACTION ####
################################

# 2d PSF values fiber 1 (sci)
# sigma_coeffs_fiber1 = np.array(, dtype=np.ndarray)
# thetas_coeffs_fiber1 = np.array(, dtype=np.ndarray)
# qs_coeffs_fiber1 = np.array(, dtype=np.ndarray)

# # 2d PSF values fiber 3 (cal)
# sigma_coeffs_fiber3 = np.array(, dtype=np.ndarray)
# thetas_coeffs_fiber3 = np.array(, dtype=np.ndarray)
# qs_coeffs_fiber3 = np.array(, dtype=np.ndarray)

read_noise = 0.0

#######################################
##### GENERATING RADIAL VELOCITIES ####
#######################################

# RV Zero point [m/s] (approx, fiber 3, Sci)
rv_zero_point = -5604.0

# LFC info
# l*f = c
# dlf + ldf=0, df=-dlf/l=-dl/l*c/l=-dl/c^2
# f = c / l
# l = c / f
f0 = cs.c / (1559.91370 * 1E-9) # freq of pump line in Hz.
df = 10.0000000 * 1E9 # spacing of peaks [Hz]

# When orientated with orders along detector rows and concave down (vertex at top)
# Top fiber is 1 (sci)
# Bottom fiber is 3 (cal)

# Approximate quadratic length solution coeffs as a starting point
wls_coeffs_fiber1 = np.array([np.array([-4.99930888e-06,  6.57331321e-02,  1.14154559e+04]), np.array([-4.96635349e-06,  6.61355741e-02,  1.14922064e+04]), np.array([-4.94002050e-06,  6.65474941e-02,  1.15731918e+04]), np.array([-4.91978434e-06,  6.69689455e-02,  1.16579459e+04]), np.array([-4.90515998e-06,  6.74001048e-02,  1.17460791e+04]), np.array([-4.89570087e-06,  6.78412473e-02,  1.18372687e+04]), np.array([-4.89099685e-06,  6.82927250e-02,  1.19312513e+04]), np.array([-4.89067214e-06,  6.87549470e-02,  1.20278154e+04]), np.array([-4.89438338e-06,  6.92283618e-02,  1.21267945e+04]), np.array([-4.90181770e-06,  6.97134412e-02,  1.22280613e+04]), np.array([-4.91269080e-06,  7.02106679e-02,  1.23315224e+04]), np.array([-4.92674517e-06,  7.07205230e-02,  1.24371131e+04]), np.array([-4.94374825e-06,  7.12434771e-02,  1.25447935e+04]), np.array([-4.96349074e-06,  7.17799821e-02,  1.26545444e+04]), np.array([-4.98578489e-06,  7.23304661e-02,  1.27663645e+04]), np.array([-5.01046292e-06,  7.28953285e-02,  1.28802668e+04]), np.array([-5.03737548e-06,  7.34749377e-02,  1.29962767e+04]), np.array([-5.06639013e-06,  7.40696303e-02,  1.31144298e+04]), np.array([-5.09738999e-06,  7.46797114e-02,  1.32347699e+04]), np.array([-5.13027239e-06,  7.53054565e-02,  1.33573479e+04]), np.array([-5.16494762e-06,  7.59471144e-02,  1.34822205e+04]), np.array([-5.20133781e-06,  7.66049115e-02,  1.36094491e+04]), np.array([-5.23937585e-06,  7.72790568e-02,  1.37390993e+04]), np.array([-5.27900446e-06,  7.79697479e-02,  1.38712404e+04]), np.array([-5.32017532e-06,  7.86771778e-02,  1.40059450e+04]), np.array([-5.36284833e-06,  7.94015420e-02,  1.41432889e+04]), np.array([-5.40699106e-06,  8.01430459e-02,  1.42833509e+04]), np.array([-5.45257819e-06,  8.09019130e-02,  1.44262128e+04]), np.array([-5.49959126e-06,  8.16783919e-02,  1.45719597e+04]), np.array([-5.54801842e-06,  8.24727647e-02,  1.47206801e+04]), np.array([-5.59785442e-06,  8.32853530e-02,  1.48724659e+04]), np.array([-5.64910080e-06,  8.41165252e-02,  1.50274127e+04]), np.array([-5.70176616e-06,  8.49667008e-02,  1.51856205e+04]), np.array([-5.75586675e-06,  8.58363550e-02,  1.53471932e+04]), np.array([-5.81142717e-06,  8.67260208e-02,  1.55122395e+04]), np.array([-5.86848138e-06,  8.76362894e-02,  1.56808726e+04]), np.array([-5.92707389e-06,  8.85678082e-02,  1.58532111e+04]), np.array([-5.98726128e-06,  8.95212760e-02,  1.60293786e+04]), np.array([-6.04911393e-06,  9.04974352e-02,  1.62095044e+04]), np.array([-6.11271815e-06,  9.14970599e-02,  1.63937234e+04]), np.array([-6.17817854e-06,  9.25209401e-02,  1.65821769e+04]), np.array([-6.24562080e-06,  9.35698607e-02,  1.67750126e+04]), np.array([-6.31519485e-06,  9.46445759e-02,  1.69723850e+04]), np.array([-6.38707844e-06,  9.57457761e-02,  1.71744563e+04]), np.array([-6.46148117e-06,  9.68740491e-02,  1.73813970e+04]), np.array([-6.53864897e-06,  9.80298331e-02,  1.75933867e+04])], dtype=np.ndarray)

wls_coeffs_fiber3 = np.array([np.array([7.20697948e-06, 3.67460556e-02, 1.12286214e+04]), np.array([5.05581638e-06, 4.24663111e-02, 1.13501810e+04]), np.array([3.22528768e-06, 4.73711982e-02, 1.14667250e+04]), np.array([1.67690770e-06, 5.15643447e-02, 1.15794400e+04]), np.array([3.75534581e-07, 5.51396921e-02, 1.16893427e+04]), np.array([-7.10802964e-07,  5.81820837e-02,  1.17972991e+04]), np.array([-1.61107682e-06,  6.07678378e-02,  1.19040438e+04]), np.array([-2.35142842e-06,  6.29653049e-02,  1.20101966e+04]), np.array([-2.95533525e-06,  6.48354095e-02,  1.21162779e+04]), np.array([-3.44377496e-06,  6.64321764e-02,  1.22227231e+04]), np.array([-3.83538672e-06,  6.78032403e-02,  1.23298942e+04]), np.array([-4.14662982e-06,  6.89903401e-02,  1.24380922e+04]), np.array([-4.39193925e-06,  7.00297962e-02,  1.25475660e+04]), np.array([-4.58387820e-06,  7.09529713e-02,  1.26585223e+04]), np.array([-4.73328709e-06,  7.17867142e-02,  1.27711326e+04]), np.array([-4.84942915e-06,  7.25537869e-02,  1.28855405e+04]), np.array([-4.94013215e-06,  7.32732739e-02,  1.30018676e+04]), np.array([-5.01192617e-06,  7.39609738e-02,  1.31202183e+04]), np.array([-5.07017706e-06,  7.46297729e-02,  1.32406847e+04]), np.array([-5.11921536e-06,  7.52900004e-02,  1.33633496e+04]), np.array([-5.16246045e-06,  7.59497651e-02,  1.34882899e+04]), np.array([-5.20253946e-06,  7.66152723e-02,  1.36155790e+04]), np.array([-5.24140082e-06,  7.72911219e-02,  1.37452888e+04]), np.array([-5.28042184e-06,  7.79805856e-02,  1.38774911e+04]), np.array([-5.32051014e-06,  7.86858642e-02,  1.40122593e+04]), np.array([-5.36219842e-06,  7.94083236e-02,  1.41496685e+04]), np.array([-5.40573212e-06,  8.01487089e-02,  1.42897968e+04]), np.array([-5.45114946e-06,  8.09073365e-02,  1.44327255e+04]), np.array([-5.49835342e-06,  8.16842630e-02,  1.45785395e+04]), np.array([-5.54717499e-06,  8.24794307e-02,  1.47273271e+04]), np.array([-5.59742715e-06,  8.32927882e-02,  1.48791809e+04]), np.array([-5.64894883e-06,  8.41243860e-02,  1.50341969e+04]), np.array([-5.70163833e-06,  8.49744455e-02,  1.51924756e+04]), np.array([-5.75547508e-06,  8.58434015e-02,  1.53541213e+04]), np.array([-5.81052925e-06,  8.67319152e-02,  1.55192427e+04]), np.array([-5.86695809e-06,  8.76408587e-02,  1.56879531e+04]), np.array([-5.92498805e-06,  8.85712681e-02,  1.58603707e+04]), np.array([-5.98488157e-06,  8.95242649e-02,  1.60366187e+04]), np.array([-6.04688747e-06,  9.05009435e-02,  1.62168261e+04]), np.array([-6.11117352e-06,  9.15022230e-02,  1.64011279e+04]), np.array([-6.17773991e-06,  9.25286629e-02,  1.65896656e+04]), np.array([-6.24631195e-06,  9.35802387e-02,  1.67825877e+04]), np.array([-6.31621058e-06,  9.46560766e-02,  1.69800498e+04]), np.array([-6.38619856e-06,  9.57541455e-02,  1.71822150e+04]), np.array([-6.45430063e-06,  9.68709018e-02,  1.73892538e+04]), np.array([-6.51759531e-06,  9.80008858e-02,  1.76013435e+04])], dtype=np.ndarray)


# LSF widths (temporary)
lsf_linear_coeffs = np.array([0.00167515, 0.08559148])
lsf_widths = np.polyval(lsf_linear_coeffs, np.arange(46))
lsf_widths = [[lw*0.5, lw, lw*1.5] for lw in lsf_widths]