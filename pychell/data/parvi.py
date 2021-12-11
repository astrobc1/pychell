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
from astropy.time import Time, TimeDelta
import astropy.units as units
from astropy.io import fits

# Barycorrpy
from barycorrpy import get_BC_vel
from barycorrpy.utc_tdb import JDUTC_to_BJDTDB

# Pychell deps
import pychell.maths as pcmath
import pychell.data.spectraldata as pcspecdata
from pychell.reduce.recipes import ReduceRecipe


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

def categorize_raw_data(data_input_path, output_path, full_flats_path=None, fiber_flats_path=None, darks_path=None, lfc_path=None, badpix_mask_file=None):

    # Defaults
    if full_flats_path is None:
        full_flats_path = data_input_path
    if fiber_flats_path is None:
        fiber_flats_path = data_input_path
    if darks_path is None:
        darks_path = data_input_path
    if lfc_path is None:
        lfc_path = data_input_path

    # Stores the data as above objects
    data = {}

    # Classify files
    all_files = glob.glob(data_input_path + '*.fits')
    lfc_files = glob.glob(lfc_path + 'LFC_*.fits')
    dark_files = glob.glob(darks_path + '*DARK*.fits')
    full_flat_files = glob.glob(full_flats_path + '*FULLFLAT*.fits')
    fiber_flat_files = glob.glob(fiber_flats_path + '*FIBERFLAT*.fits') + glob.glob(fiber_flats_path + '*FIBREFLAT*.fits')
    sci_files = list(set(all_files) - set(lfc_files) - set(dark_files) - set(full_flat_files) - set(fiber_flat_files))

    # Create Echellograms from raw data
    data['science'] = [pcspecdata.RawEchellogram(input_file=f, spectrograph=spectrograph) for f in sci_files]
    data['fiber_flats'] = [pcspecdata.RawEchellogram(input_file=f, spectrograph=spectrograph) for f in fiber_flat_files]
    data['darks'] = [pcspecdata.RawEchellogram(input_file=f, spectrograph=spectrograph) for f in dark_files]
    data['flats'] = [pcspecdata.RawEchellogram(input_file=f, spectrograph=spectrograph) for f in full_flat_files]
    data['lfc'] = [pcspecdata.RawEchellogram(input_file=f, spectrograph=spectrograph) for f in lfc_files]

    # Only get latest cals
    dark_group = get_latest_darks(data)
    full_flat_group = get_latest_full_flats(data)
    fiber_flat_group1 = get_latest_fiber_flats(data, fiber=1)
    fiber_flat_group3 = get_latest_fiber_flats(data, fiber=3)

    # Master Darks
    if len(dark_files) > 0:
        data['master_darks'] = [pcspecdata.MasterCal(dark_group, output_path + "calib" + os.sep)]

    # Master Flats
    if len(full_flat_files) > 0:
        data['master_flats'] = [pcspecdata.MasterCal(full_flat_group, output_path + "calib" + os.sep)]
    
    # Master fiber flats
    data['master_fiber_flats'] = [pcspecdata.MasterCal(fiber_flat_group1, output_path + "calib" + os.sep), pcspecdata.MasterCal(fiber_flat_group3, output_path + "calib" + os.sep)]

    # Order maps
    data['order_maps'] = data['master_fiber_flats']

    # Which to extract
    data['extract'] = data['science'] + data['master_fiber_flats'] + data['lfc']

    # Pair order maps for the spectra to extract
    for d in data['extract']:
        pair_order_maps(d, data['order_maps'])
        if len(dark_files) > 0:
            pair_master_dark(d, data['master_darks'])
        if len(full_flat_files) > 0:
            pair_master_flat(d, data['master_flats'])

    # Pair darks with full frame flats
    if len(full_flat_files) > 0 and len(dark_files) > 0:
        for flat in data['flats']:
            pair_master_dark(flat, data['master_darks'])

    # Bad pixel mask (only one, load into memory)
    data['badpix_mask'] = 1 - fits.open(badpix_mask_file)[0].data.astype(float)

    return data

def get_latest_darks(data):
    dates = [parse_utdate(d) for d in data["darks"]]
    dates_unq = np.unique(dates)
    dates_unq = np.sort(dates_unq)
    date_use = dates_unq[-1]
    group = [d for i, d in enumerate(data["darks"]) if dates[i] == date_use]
    return group

def get_latest_full_flats(data):
    dates = [parse_utdate(d) for d in data["flats"]]
    dates_unq = np.unique(dates)
    dates_unq = np.sort(dates_unq)
    date_use = dates_unq[-1]
    group = [d for i, d in enumerate(data["flats"]) if dates[i] == date_use]
    return group

def get_latest_fiber_flats(data, fiber=1):
    dates = [parse_utdate(d) for d in data["fiber_flats"]]
    fibers = [parse_fiber_nums(d) for d in data["fiber_flats"]]
    dates_unq = np.unique(dates)
    dates_unq = np.sort(dates_unq)
    date_use = dates_unq[-1]
    group = [d for i, d in enumerate(data["fiber_flats"]) if dates[i] == date_use and fibers[i] == fiber]
    return group


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
    try:
        parse_utdate(data)
    except:
        print(f"No ut date found for {data}")

    try:
        parse_sky_coord(data)
    except:
        print(f"No sky coord found for {data}")
    parse_exposure_start_time(data)
    parse_object(data)
    parse_itime(data)
    
    return data.header

def parse_image(data, scale_to_itime=True):
    image = fits.open(data.input_file, do_not_scale_image_data=True)[0].data.astype(float)
    if "master" in data.base_input_file:
        return image
    else:
        image = image.T
        if scale_to_itime:
            image *= parse_itime(data)
            return image
        else:
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
    jdlocstart = parse_exposure_start_time(data) + TimeDelta(utc_offset*3600, format='sec')
    y, m, d = str(jdlocstart.datetime.year), str(jdlocstart.datetime.month), str(jdlocstart.datetime.day)
    if len(m) == 1:
        m = f"0{m}"
    if len(d) == 1:
        d = f"0{d}"
    utdate = [str(y), str(m), str(d)]
    utdate = "".join(utdate)
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


#######################
#### GROUPING CALS ####
#######################

def group_darks(darks):
    return [darks]

def group_flats(flats):
    return [flats]

def gen_master_calib_filename(master_cal):
    fname0 = master_cal.group[0].base_input_file.lower()
    if "dark" in fname0:
        return f"master_dark_{master_cal.group[0].utdate}{master_cal.group[0].itime}s.fits"
    elif "fiberflat" in fname0 or "fibreflat" in fname0:
        return f"master_fiberflat_{master_cal.group[0].utdate}_fiber{parse_fiber_nums(master_cal.group[0])}.fits"
    elif "fullflat" in fname0:
        return f"master_fullflat_{master_cal.group[0].utdate}.fits"
    elif "lfc" in fname0:
        return f"master_lfc_{master_cal.group[0].utdate}.fits"
    else:
        return f"master_calib_{master_cal.group[0].utdate}.fits"

def gen_master_calib_header(master_cal):
    return copy.deepcopy(master_cal.group[0].header)


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
sigma_coeffs_fiber1 = np.array([np.array([ 2.68986911e-07, -2.21371834e-04,  1.62048242e+00]),
       np.array([ 4.30803466e-07, -5.23453972e-04,  1.80873237e+00]),
       np.array([ 2.40330465e-07, -1.45705064e-04,  1.58855819e+00]),
       np.array([9.93724618e-08, 2.14636269e-07, 1.54056591e+00]),
       np.array([9.44650308e-08, 8.27711890e-05, 1.38450065e+00]),
       np.array([1.15600740e-07, 4.54628824e-05, 1.39736087e+00]),
       np.array([ 1.36218329e-07, -2.32376282e-05,  1.45646093e+00]),
       np.array([9.64764365e-08, 4.00637684e-05, 1.45665065e+00]),
       np.array([ 1.44658611e-07, -4.50053231e-05,  1.51113069e+00]),
       np.array([ 1.86751631e-07, -1.30896401e-04,  1.57755831e+00]),
       np.array([ 1.98540651e-07, -1.70679654e-04,  1.61689634e+00]),
       np.array([ 1.33437970e-07, -5.08106218e-05,  1.58475430e+00]),
       np.array([ 1.49913021e-07, -5.23376171e-05,  1.58238211e+00]),
       np.array([ 1.48372399e-07, -3.28201516e-05,  1.58576763e+00]),
       np.array([-1.05838425e-07,  3.58495133e-04,  1.61462805e+00]),
       np.array([8.98706817e-08, 6.58748299e-05, 1.60561977e+00]),
       np.array([7.65496907e-08, 7.58176285e-05, 1.61655196e+00]),
       np.array([1.26380376e-07, 2.50452835e-06, 1.67152882e+00]),
       np.array([9.13719087e-08, 7.03549281e-05, 1.70532144e+00]),
       np.array([1.24374285e-07, 2.69856939e-05, 1.84015864e+00]),
       np.array([1.38580861e-07, 1.24034096e-05, 1.89864900e+00])], dtype=np.ndarray)

q_coeffs_fiber1 = np.array([np.array([ 8.94993664e-08, -3.20059415e-04,  7.97740336e-01]),
       np.array([ 8.38733107e-08, -2.89210225e-04,  7.62255118e-01]),
       np.array([ 1.28920178e-07, -3.92561753e-04,  8.19761528e-01]),
       np.array([ 7.89358672e-08, -2.97021332e-04,  7.86529869e-01]),
       np.array([ 7.25688631e-08, -3.02269797e-04,  8.00805653e-01]),
       np.array([ 1.32694971e-07, -4.04525811e-04,  8.25495446e-01]),
       np.array([ 1.39918679e-07, -3.77843286e-04,  7.87431458e-01]),
       np.array([ 3.58612270e-08, -1.53556004e-04,  6.84387258e-01]),
       np.array([ 2.24269843e-08, -1.48593195e-04,  6.98198157e-01]),
       np.array([ 4.15834113e-08, -1.80849611e-04,  7.07522965e-01]),
       np.array([-1.48791371e-08, -1.66429985e-04,  7.13326854e-01]),
       np.array([ 1.50242489e-07, -3.68407115e-04,  7.67013528e-01]),
       np.array([ 1.43332271e-07, -3.44327028e-04,  7.54940232e-01]),
       np.array([ 1.23235496e-07, -3.07248658e-04,  7.55636648e-01]),
       np.array([ 1.65587309e-07, -4.04185585e-04,  8.08619093e-01]),
       np.array([ 8.80399936e-08, -2.12431257e-04,  7.01332968e-01]),
       np.array([ 8.16763445e-08, -1.93522978e-04,  7.01152982e-01]),
       np.array([ 8.95938293e-09, -6.35913125e-05,  6.61521585e-01]),
       np.array([ 5.61546370e-08, -1.86291868e-04,  7.22930646e-01]),
       np.array([ 6.53123797e-08, -1.51719828e-04,  7.03369417e-01]),
       np.array([ 6.18484254e-08, -1.35954787e-04,  7.00907514e-01])], dtype=np.ndarray)

theta_coeffs_fiber1 = np.array([np.array([-3.22168476e-11,  1.72271252e-07, -3.46907422e-04,  1.53593693e+00]),
       np.array([-1.06914887e-10,  4.45445781e-07, -6.37628487e-04,  1.44805445e+00]),
       np.array([ 1.54875756e-11, -1.61173820e-08, -5.60985934e-05,  1.17889062e+00]),
       np.array([ 1.22098218e-10, -4.05939690e-07,  2.38987148e-04,  1.12780102e+00]),
       np.array([ 1.35821429e-10, -5.38933830e-07,  6.03665171e-04,  8.05144867e-01]),
       np.array([ 2.52749929e-10, -9.75618309e-07,  1.13730812e-03,  5.55066844e-01]),
       np.array([ 3.54649911e-10, -1.34445673e-06,  1.52959726e-03,  4.23850646e-01]),
       np.array([ 6.91512181e-11, -3.39138305e-07,  5.03846507e-04,  6.77477138e-01]),
       np.array([ 9.15745762e-11, -4.63567301e-07,  6.92788196e-04,  5.88540729e-01]),
       np.array([ 1.12801388e-10, -5.56210976e-07,  8.51860784e-04,  4.98105746e-01]),
       np.array([ 1.50253653e-10, -6.96914097e-07,  1.03150212e-03,  4.04811512e-01]),
       np.array([ 2.69382550e-10, -1.05609689e-06,  1.28040602e-03,  4.37280555e-01]),
       np.array([ 2.35667037e-10, -9.36130060e-07,  1.15293015e-03,  4.73141316e-01]),
       np.array([ 2.36623113e-10, -9.07605389e-07,  1.10198715e-03,  5.09230369e-01]),
       np.array([ 4.29360698e-10, -1.47520808e-06,  1.33817462e-03,  8.20557714e-01]),
       np.array([ 2.33777632e-10, -8.55957761e-07,  9.26418486e-04,  6.37750485e-01]),
       np.array([ 1.52454313e-10, -5.50010136e-07,  6.19188705e-04,  6.95534164e-01]),
       np.array([ 9.18834584e-12, -6.65789505e-08,  1.70552482e-04,  8.02883431e-01]),
       np.array([ 1.03533981e-10, -4.10145737e-07,  4.89571959e-04,  8.27029241e-01]),
       np.array([-7.48995412e-13,  1.07834799e-07, -2.96749832e-04,  1.09985808e+00]),
       np.array([-5.60297787e-11,  3.49206034e-07, -6.29501952e-04,  1.26657330e+00])], dtype=np.ndarray)


# # 2d PSF values fiber 3 (cal)
# sigma_coeffs_fiber3 = np.array(, dtype=np.ndarray)
# thetas_coeffs_fiber3 = np.array(, dtype=np.ndarray)
# qs_coeffs_fiber3 = np.array(, dtype=np.ndarray)

read_noise = 0.0 # Needs updated

class PARVIReduceRecipe(ReduceRecipe):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, data_input_path, output_path, full_flats_path=None, fiber_flats_path=None, darks_path=None, lfc_path=None, badpix_mask_file=None, do_bias=False, do_dark=True, do_flat=True, flat_percentile=0.5, mask_left=50, mask_right=50, mask_top=10, mask_bottom=10, tracer=None, extractor=None, n_cores=1, tilts=None):

        # Super init
        super().__init__(spectrograph="PARVI", data_input_path=data_input_path, output_path=output_path, do_bias=do_bias, do_dark=do_dark, do_flat=do_flat, flat_percentile=flat_percentile, mask_left=mask_left, mask_right=mask_right, mask_top=mask_top, mask_bottom=mask_bottom, tracer=tracer, extractor=extractor, n_cores=n_cores)
        
        # Store additional PARVI related params
        self.full_flats_path = full_flats_path
        self.fiber_flats_path = fiber_flats_path
        self.darks_path = darks_path
        self.lfc_path = lfc_path
        self.badpix_mask_file = badpix_mask_file

        # Init the data
        self.init_data()

    def init_data(self):
        """Initialize the data by calling self.categorize_raw_data.
        """
        
        # Identify what's what.
        print("Categorizing Data ...", flush=True)
        self.data = self.spec_module.categorize_raw_data(self.data_input_path, self.output_path, self.full_flats_path, self.fiber_flats_path, self.darks_path, self.lfc_path, self.badpix_mask_file)

        # Print reduction summary
        self.print_reduction_summary()

    @staticmethod
    def prep_post_reduction_products(path):

        # Parse folder
        all_files = glob.glob(f"{path}*_reduced.fits")
        lfc_files = glob.glob(f"{path}*LFC*_reduced.fits")
        fiber_flat_files = glob.glob(f"{path}*FIBERFLAT*_reduced.fits") + glob.glob(f"{path}*FIBREFLAT*_reduced.fits")
        sci_files = list(set(all_files) - set(lfc_files) - set(fiber_flat_files))

        # Create temporary objects to parse header info
        scis = [pcspecdata.RawEchellogram(f, spectrograph="PARVI") for f in sci_files]
        lfc_cals = [pcspecdata.RawEchellogram(f, spectrograph="PARVI") for f in lfc_files]
        fiber_flats = [pcspecdata.RawEchellogram(f, spectrograph="PARVI") for f in fiber_flat_files]

        # Parse times of lfc cals
        times_lfc_cal = np.array([compute_exposure_midpoint(d) for d in lfc_cals], dtype=float)
        ss = np.argsort(times_lfc_cal)
        lfc_files = np.array(lfc_files)
        lfc_files = lfc_files[ss]
        times_lfc_cal = times_lfc_cal[ss]
        lfc_cals = [lfc_cals[ss[i]] for i in range(len(lfc_cals))]
        
        # Parse times of sci exposures
        times_sci = np.array([compute_exposure_midpoint(d) for d in scis], dtype=float)
        ss = np.argsort(times_sci)
        sci_files = np.array(sci_files)
        sci_files = sci_files[ss]
        times_sci = times_sci[ss]
        scis = [scis[ss[i]] for i in range(len(scis))]

        # Initialize arrays for lfc spectra
        n_orders, nx = 21, 2048
        lfc_cal_scifiber = np.full((nx, n_orders, len(lfc_files)), np.nan)
        lfc_cal_calfiber = np.full((nx, n_orders, len(lfc_files)), np.nan)
        lfc_sci_calfiber = np.full((nx, n_orders, len(sci_files)), np.nan)

        # Load in 1d lfc spectra from cal (dual fiber) files
        for i in range(len(lfc_files)):
            lfc_data = fits.open(lfc_files[i])[0].data
            lfc_cal_scifiber[:, :, i] = lfc_data[:, 0, :, 0].T
            lfc_cal_calfiber[:, :, i] = lfc_data[:, 1, :, 0].T

        # Load in 1d lfc spectra from sci (only cal fiber) files
        for i in range(len(sci_files)):
            lfc_sci_calfiber[:, :, i] = fits.open(sci_files[i])[0].data[:, 1, :, 0].T

        # Initialize fiber flat arrays
        fiber_flat_data = np.full((nx, n_orders, len(fiber_flat_files)), np.nan)

        # Load fiber flat data
        for i in range(len(fiber_flat_files)):
            fiber_flat_data[:, :, i] = fits.open(fiber_flat_files[i])[0].data[:, 0, :, 0].T

        return scis, times_sci, lfc_sci_calfiber, lfc_cals, times_lfc_cal, lfc_cal_scifiber, lfc_cal_calfiber, fiber_flats, fiber_flat_data

    # Outputs
    def save_final(self, sci_file, wls_sciobs_scifiber, wls_sciobs_calfiber, continuum_corrections=None):

        # Load sci file
        sci_data = fits.open(sci_file)[0].data
        header = fits.open(sci_file)[0].header
        n_orders = sci_data.shape[0]
        nx = sci_data.shape[2]

        # Initiate output array
        # LAST DIMENSION:
        # 0: sci_wave
        # 1: sci_flux
        # 2: sci_flux_unc
        # 3: sci_badpix
        # 4: sci_continuum
        # 5: simult_lfc_wave
        # 6: simult_lfc_flux
        # 7: simult_lfc_flux_unc
        # 8: simult_lfc_badpix
        data_out = np.full((n_orders, nx, 9), np.nan)

        # Fill wls
        data_out[:, :, 0] = wls_sciobs_scifiber.T

        # Fill science
        data_out[:, :, 1:4] = sci_data[:, 0, :, :]

        # Fill fiber flat
        if continuum_corrections is not None:
            data_out[:, :, 4] = continuum_corrections.T
        else:
            data_out[:, :, 4] = 1.0

        # Fill simult LFC
        data_out[:, :, 5] = wls_sciobs_calfiber.T
        data_out[:, :, 6:] = sci_data[:, 1, :, :]

        # Save
        hdul = fits.HDUList([fits.PrimaryHDU(data_out, header=header)])
        fname = f"{sci_file[:-5]}_calibrated.fits"
        hdul.writeto(fname, overwrite=True, output_verify='ignore')
    

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
lfc_f0 = cs.c / (1559.91370 * 1E-9) # freq of pump line in Hz.
lfc_df = 10.0000000 * 1E9 # spacing of peaks [Hz]

# When orientated with orders along detector rows and concave down (vertex at top)
# Top fiber is 1 (sci)
# Bottom fiber is 3 (cal)

# Approximate quadratic length solution coeffs as a starting point
wls_coeffs_fiber1 = np.array([np.array([-4.99930888e-06,  6.57331321e-02,  1.14154559e+04]), np.array([-4.96635349e-06,  6.61355741e-02,  1.14922064e+04]), np.array([-4.94002050e-06,  6.65474941e-02,  1.15731918e+04]), np.array([-4.91978434e-06,  6.69689455e-02,  1.16579459e+04]), np.array([-4.90515998e-06,  6.74001048e-02,  1.17460791e+04]), np.array([-4.89570087e-06,  6.78412473e-02,  1.18372687e+04]), np.array([-4.89099685e-06,  6.82927250e-02,  1.19312513e+04]), np.array([-4.89067214e-06,  6.87549470e-02,  1.20278154e+04]), np.array([-4.89438338e-06,  6.92283618e-02,  1.21267945e+04]), np.array([-4.90181770e-06,  6.97134412e-02,  1.22280613e+04]), np.array([-4.91269080e-06,  7.02106679e-02,  1.23315224e+04]), np.array([-4.92674517e-06,  7.07205230e-02,  1.24371131e+04]), np.array([-4.94374825e-06,  7.12434771e-02,  1.25447935e+04]), np.array([-4.96349074e-06,  7.17799821e-02,  1.26545444e+04]), np.array([-4.98578489e-06,  7.23304661e-02,  1.27663645e+04]), np.array([-5.01046292e-06,  7.28953285e-02,  1.28802668e+04]), np.array([-5.03737548e-06,  7.34749377e-02,  1.29962767e+04]), np.array([-5.06639013e-06,  7.40696303e-02,  1.31144298e+04]), np.array([-5.09738999e-06,  7.46797114e-02,  1.32347699e+04]), np.array([-5.13027239e-06,  7.53054565e-02,  1.33573479e+04]), np.array([-5.16494762e-06,  7.59471144e-02,  1.34822205e+04]), np.array([-5.20133781e-06,  7.66049115e-02,  1.36094491e+04]), np.array([-5.23937585e-06,  7.72790568e-02,  1.37390993e+04]), np.array([-5.27900446e-06,  7.79697479e-02,  1.38712404e+04]), np.array([-5.32017532e-06,  7.86771778e-02,  1.40059450e+04]), np.array([-5.36284833e-06,  7.94015420e-02,  1.41432889e+04]), np.array([-5.40699106e-06,  8.01430459e-02,  1.42833509e+04]), np.array([-5.45257819e-06,  8.09019130e-02,  1.44262128e+04]), np.array([-5.49959126e-06,  8.16783919e-02,  1.45719597e+04]), np.array([-5.54801842e-06,  8.24727647e-02,  1.47206801e+04]), np.array([-5.59785442e-06,  8.32853530e-02,  1.48724659e+04]), np.array([-5.64910080e-06,  8.41165252e-02,  1.50274127e+04]), np.array([-5.70176616e-06,  8.49667008e-02,  1.51856205e+04]), np.array([-5.75586675e-06,  8.58363550e-02,  1.53471932e+04]), np.array([-5.81142717e-06,  8.67260208e-02,  1.55122395e+04]), np.array([-5.86848138e-06,  8.76362894e-02,  1.56808726e+04]), np.array([-5.92707389e-06,  8.85678082e-02,  1.58532111e+04]), np.array([-5.98726128e-06,  8.95212760e-02,  1.60293786e+04]), np.array([-6.04911393e-06,  9.04974352e-02,  1.62095044e+04]), np.array([-6.11271815e-06,  9.14970599e-02,  1.63937234e+04]), np.array([-6.17817854e-06,  9.25209401e-02,  1.65821769e+04]), np.array([-6.24562080e-06,  9.35698607e-02,  1.67750126e+04]), np.array([-6.31519485e-06,  9.46445759e-02,  1.69723850e+04]), np.array([-6.38707844e-06,  9.57457761e-02,  1.71744563e+04]), np.array([-6.46148117e-06,  9.68740491e-02,  1.73813970e+04]), np.array([-6.53864897e-06,  9.80298331e-02,  1.75933867e+04])], dtype=np.ndarray)

wls_coeffs_fiber3 = np.array([np.array([7.20697948e-06, 3.67460556e-02, 1.12286214e+04]), np.array([5.05581638e-06, 4.24663111e-02, 1.13501810e+04]), np.array([3.22528768e-06, 4.73711982e-02, 1.14667250e+04]), np.array([1.67690770e-06, 5.15643447e-02, 1.15794400e+04]), np.array([3.75534581e-07, 5.51396921e-02, 1.16893427e+04]), np.array([-7.10802964e-07,  5.81820837e-02,  1.17972991e+04]), np.array([-1.61107682e-06,  6.07678378e-02,  1.19040438e+04]), np.array([-2.35142842e-06,  6.29653049e-02,  1.20101966e+04]), np.array([-2.95533525e-06,  6.48354095e-02,  1.21162779e+04]), np.array([-3.44377496e-06,  6.64321764e-02,  1.22227231e+04]), np.array([-3.83538672e-06,  6.78032403e-02,  1.23298942e+04]), np.array([-4.14662982e-06,  6.89903401e-02,  1.24380922e+04]), np.array([-4.39193925e-06,  7.00297962e-02,  1.25475660e+04]), np.array([-4.58387820e-06,  7.09529713e-02,  1.26585223e+04]), np.array([-4.73328709e-06,  7.17867142e-02,  1.27711326e+04]), np.array([-4.84942915e-06,  7.25537869e-02,  1.28855405e+04]), np.array([-4.94013215e-06,  7.32732739e-02,  1.30018676e+04]), np.array([-5.01192617e-06,  7.39609738e-02,  1.31202183e+04]), np.array([-5.07017706e-06,  7.46297729e-02,  1.32406847e+04]), np.array([-5.11921536e-06,  7.52900004e-02,  1.33633496e+04]), np.array([-5.16246045e-06,  7.59497651e-02,  1.34882899e+04]), np.array([-5.20253946e-06,  7.66152723e-02,  1.36155790e+04]), np.array([-5.24140082e-06,  7.72911219e-02,  1.37452888e+04]), np.array([-5.28042184e-06,  7.79805856e-02,  1.38774911e+04]), np.array([-5.32051014e-06,  7.86858642e-02,  1.40122593e+04]), np.array([-5.36219842e-06,  7.94083236e-02,  1.41496685e+04]), np.array([-5.40573212e-06,  8.01487089e-02,  1.42897968e+04]), np.array([-5.45114946e-06,  8.09073365e-02,  1.44327255e+04]), np.array([-5.49835342e-06,  8.16842630e-02,  1.45785395e+04]), np.array([-5.54717499e-06,  8.24794307e-02,  1.47273271e+04]), np.array([-5.59742715e-06,  8.32927882e-02,  1.48791809e+04]), np.array([-5.64894883e-06,  8.41243860e-02,  1.50341969e+04]), np.array([-5.70163833e-06,  8.49744455e-02,  1.51924756e+04]), np.array([-5.75547508e-06,  8.58434015e-02,  1.53541213e+04]), np.array([-5.81052925e-06,  8.67319152e-02,  1.55192427e+04]), np.array([-5.86695809e-06,  8.76408587e-02,  1.56879531e+04]), np.array([-5.92498805e-06,  8.85712681e-02,  1.58603707e+04]), np.array([-5.98488157e-06,  8.95242649e-02,  1.60366187e+04]), np.array([-6.04688747e-06,  9.05009435e-02,  1.62168261e+04]), np.array([-6.11117352e-06,  9.15022230e-02,  1.64011279e+04]), np.array([-6.17773991e-06,  9.25286629e-02,  1.65896656e+04]), np.array([-6.24631195e-06,  9.35802387e-02,  1.67825877e+04]), np.array([-6.31621058e-06,  9.46560766e-02,  1.69800498e+04]), np.array([-6.38619856e-06,  9.57541455e-02,  1.71822150e+04]), np.array([-6.45430063e-06,  9.68709018e-02,  1.73892538e+04]), np.array([-6.51759531e-06,  9.80008858e-02,  1.76013435e+04])], dtype=np.ndarray)