# Base Python
import os
import copy
import glob
import sys

# Maths
import numpy as np
import scipy.constants as cs

# Astropy
from astropy.io import fits
from astropy.coordinates import EarthLocation
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
import astropy.units as units

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
    "site" : EarthLocation.of_site("Palomar")
}

echelle_orders = [84, 129]

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
    fiber_flat_group1 = get_latest_fiber_flats(data, fiber='1')
    fiber_flat_group3 = get_latest_fiber_flats(data, fiber='3')

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

def get_latest_fiber_flats(data, fiber):
    dates = [parse_utdate(d) for d in data["fiber_flats"]]
    fibers = [parse_fiber_nums(d)[0] for d in data["fiber_flats"]]
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
    fibers_sci = parse_fiber_nums(data)
    fibers_order_maps = np.array([parse_fiber_nums(order_map)[0] for order_map in order_maps], dtype='<U10')
    order_maps_out = []
    for fiber in fibers_sci:
        k = np.where(fibers_order_maps == fiber)[0]
        if len(k) == 0:
            raise ValueError(f"No fiber flat corresponding to {data}")
        else:
            order_maps_out.append(order_maps[k[0]])
    data.order_maps = order_maps_out

def parse_image_num(data):
    return 1

def parse_object(data):
    data.object = data.header["OBJECT"]
    return data.object
    
def parse_utdate(data):
    utc_offset = -8
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
    if "fiberflat" in data.base_input_file.lower() or "fibreflat" in data.base_input_file.lower():
        return [str(data.header["FIBER"])]
    else:
        return ["1", "3"]

def compute_echelle_order_num(data=None, order_num=None):
    if order_num is None:
        order_num = data.order_num
    m = np.polyval([-1, 130], order_num)
    return m

def parse_spec1d(data):
    fits_data = fits.open(data.input_file)
    fits_data.verify('fix')
    data.header = fits_data[0].header
    oi = (echelle_orders[1] - echelle_orders[0]) - (data.order_num - echelle_orders[0])
    data.wave = fits_data[0].data[oi, :, 0]
    #data.flux = fits_data[0].data[oi, :, 1] / fits_data[0].data[oi, :, 4] # continuum correction from fiber flat
    data.flux = fits_data[0].data[oi, :, 1]
    #breakpoint()
    data.flux_unc = fits_data[0].data[oi, :, 2]
    data.mask = fits_data[0].data[oi, :, 3]

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
        return f"master_fiberflat_{master_cal.group[0].utdate}_fiber{parse_fiber_nums(master_cal.group[0])[0]}.fits"
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
        pcoeffs = wls_coeffs_fiber1[order_num]
    else:
        pcoeffs = wls_coeffs_fiber3[order_num]
    wls = np.polyval(pcoeffs, np.arange(2048).astype(float))
    return wls


################################
#### REDUCTION / EXTRACTION ####
################################

# 2d PSF Chebyshev coeffs fiber 1 (sci)
psf_fiber1_sigma_cc_coeffs = np.array([[10817.09583765, -17107.01816095, 7967.16476104, -1811.53741748],
                                [-18612.91229572, 29436.74986453, -13706.5152384, 3114.74792951],
                                [9548.27002104, -15096.6068257, 7025.28781907, -1593.86949674],
                                [-2774.265692, 4383.52793171, -2036.75971941, 460.22238848]])

psf_fiber1_q_cc_coeffs = np.array([[-3139.67583903, 4982.58222991, -2335.59395504, 541.54013115],
                            [5451.17067582, -8645.92564126, 4049.37069305, -936.34868835],
                            [-3023.09460623, 4791.53953884, -2241.25096312, 516.0933734 ],
                            [1042.48068141, -1650.82267482, 770.83132352, -176.50370795]])

psf_fiber1_theta_cc_coeffs = np.array([[-466.35560876, 721.54578343, -317.54563072, 64.13745968],
                                [-2780.01076985, 4430.65003473, -2099.62461325, 494.07802169],
                                [ 2941.86932764, -4677.56860455, 2205.10624142, -514.58119973],
                                [-3118.18048152, 4937.4176478, -2306.00985419, 526.88079383]])



# 2d PSF Chebyshev coeffs fiber 3 (cal)
psf_fiber3_sigma_cc_coeffs = np.array([[4173.78757047, -6634.3759457, 3126.71874365, -733.5217621],
                                       [-7870.64107668, 12510.48265612, -5891.31058062, 1379.60205539],
                                       [ 3734.51902302, -5940.18282238, 2802.16442712, -659.07742754],
                                       [-1153.04124161, 1834.24697739, -865.60689009, 203.70287094]])

psf_fiber3_q_cc_coeffs = np.array([[818.83147371, -1288.47172767, 595.50495886, -131.23586891],
                                   [-1472.88232812, 2320.12939078, -1073.28662361, 237.5627195],
                                   [869.43234596, -1370.04895676, 633.84693128, -140.56349015],
                                   [-168.04075282, 263.69502352, -120.90625017, 26.05078767]])

psf_fiber3_theta_cc_coeffs = np.array([[-4246.43437619, 6728.54941002, -3143.75201881, 722.73701936],
                                       [6036.75325781, -9557.40480272, 4459.08715798, -1021.00456641],
                                       [-2358.72363464, 3731.87798627, -1738.49428916, 396.55550374],
                                       [269.08091673, -423.29234646, 194.52566234, -42.84035417]])


read_noise = 0.0 # Needs updated
gain = 1.0 # Needs updated

class PARVIReduceRecipe(ReduceRecipe):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, data_input_path, output_path,
                 full_flats_path=None, fiber_flats_path=None, darks_path=None, lfc_path=None, badpix_mask_file=None,
                 do_bias=False, do_dark=True, do_flat=True, flat_percentile=0.5,
                 xrange=[49, 1997],
                 poly_mask_bottom=np.array([-6.560e-05,  1.524e-01, -4.680e+01]),
                 poly_mask_top=np.array([-6.03508772e-05,  1.23052632e-01,  1.97529825e+03]),
                 tracer=None, extractor=None,
                 n_cores=1):

        # Super init
        super().__init__(spectrograph="PARVI",
                         data_input_path=data_input_path, output_path=output_path,
                         do_bias=do_bias, do_dark=do_dark, do_flat=do_flat, flat_percentile=flat_percentile,
                         xrange=xrange, poly_mask_top=poly_mask_top, poly_mask_bottom=poly_mask_bottom,
                         tracer=tracer, extractor=extractor, n_cores=n_cores)
        
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
        n_orders, nx = 46, 2048
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
wls_coeffs_fiber1 = {
    84: [-6.53864897e-06, 0.0980298331, 17593.3867],
    85: [-6.46148117e-06, 0.0968740491, 17381.397],
    86: [-6.38707844e-06, 0.0957457761, 17174.4563],
    87: [-6.31519485e-06, 0.0946445759, 16972.385],
    88: [-6.2456208e-06, 0.0935698607, 16775.0126],
    89: [-6.17817854e-06, 0.0925209401, 16582.1769],
    90: [-6.11271815e-06, 0.0914970599, 16393.7234],
    91: [-6.04911393e-06, 0.0904974352, 16209.5044],
    92: [-5.98726128e-06, 0.089521276, 16029.3786],
    93: [-5.92707389e-06, 0.0885678082, 15853.2111],
    94: [-5.86848138e-06, 0.0876362894, 15680.8726],
    95: [-5.81142717e-06, 0.0867260208, 15512.2395],
    96: [-5.75586675e-06, 0.085836355, 15347.1932],
    97: [-5.70176616e-06, 0.0849667008, 15185.6205],
    98: [-5.6491008e-06, 0.0841165252, 15027.4127],
    99: [-5.59785442e-06, 0.083285353, 14872.4659],
    100: [-5.54801842e-06, 0.0824727647, 14720.6801],
    101: [-5.49959126e-06, 0.0816783919, 14571.9597],
    102: [-5.45257819e-06, 0.080901913, 14426.2128],
    103: [-5.40699106e-06, 0.0801430459, 14283.3509],
    104: [-5.36284833e-06, 0.079401542, 14143.2889],
    105: [-5.32017532e-06, 0.0786771778, 14005.945],
    106: [-5.27900446e-06, 0.0779697479, 13871.2404],
    107: [-5.23937585e-06, 0.0772790568, 13739.0993],
    108: [-5.20133781e-06, 0.0766049115, 13609.4491],
    109: [-5.16494762e-06, 0.0759471144, 13482.2205],
    110: [-5.13027239e-06, 0.0753054565, 13357.3479],
    111: [-5.09738999e-06, 0.0746797114, 13234.7699],
    112: [-5.06639013e-06, 0.0740696303, 13114.4298],
    113: [-5.03737548e-06, 0.0734749377, 12996.2767],
    114: [-5.01046292e-06, 0.0728953285, 12880.2668],
    115: [-4.98578489e-06, 0.0723304661, 12766.3645],
    116: [-4.96349074e-06, 0.0717799821, 12654.5444],
    117: [-4.94374825e-06, 0.0712434771, 12544.7935],
    118: [-4.92674517e-06, 0.070720523, 12437.1131],
    119: [-4.9126908e-06, 0.0702106679, 12331.5224],
    120: [-4.9018177e-06, 0.0697134412, 12228.0613],
    121: [-4.89438338e-06, 0.0692283618, 12126.7945],
    122: [-4.89067214e-06, 0.068754947, 12027.8154],
    123: [-4.89099685e-06, 0.068292725, 11931.2513],
    124: [-4.89570087e-06, 0.0678412473, 11837.2687],
    125: [-4.90515998e-06, 0.0674001048, 11746.0791],
    126: [-4.91978434e-06, 0.0669689455, 11657.9459],
    127: [-4.9400205e-06, 0.0665474941, 11573.1918],
    128: [-4.96635349e-06, 0.0661355741, 11492.2064],
    129: [-4.99930888e-06, 0.0657331321, 11415.4559]
}

wls_coeffs_fiber3 = {
    84: [-6.51759531e-06, 0.0980008858, 17601.3435],
    85: [-6.45430063e-06, 0.0968709018, 17389.2538],
    86: [-6.38619856e-06, 0.0957541455, 17182.215],
    87: [-6.31621058e-06, 0.0946560766, 16980.0498],
    88: [-6.24631195e-06, 0.0935802387, 16782.5877],
    89: [-6.17773991e-06, 0.0925286629, 16589.6656],
    90: [-6.11117352e-06, 0.091502223, 16401.1279],
    91: [-6.04688747e-06, 0.0905009435, 16216.8261],
    92: [-5.98488157e-06, 0.0895242649, 16036.6187],
    93: [-5.92498805e-06, 0.0885712681, 15860.3707],
    94: [-5.86695809e-06, 0.0876408587, 15687.9531],
    95: [-5.81052925e-06, 0.0867319152, 15519.2427],
    96: [-5.75547508e-06, 0.0858434015, 15354.1213],
    97: [-5.70163833e-06, 0.0849744455, 15192.4756],
    98: [-5.64894883e-06, 0.084124386, 15034.1969],
    99: [-5.59742715e-06, 0.0832927882, 14879.1809],
    100: [-5.54717499e-06, 0.0824794307, 14727.3271],
    101: [-5.49835342e-06, 0.081684263, 14578.5395],
    102: [-5.45114946e-06, 0.0809073365, 14432.7255],
    103: [-5.40573212e-06, 0.0801487089, 14289.7968],
    104: [-5.36219842e-06, 0.0794083236, 14149.6685],
    105: [-5.32051014e-06, 0.0786858642, 14012.2593],
    106: [-5.28042184e-06, 0.0779805856, 13877.4911],
    107: [-5.24140082e-06, 0.0772911219, 13745.2888],
    108: [-5.20253946e-06, 0.0766152723, 13615.579],
    109: [-5.16246045e-06, 0.0759497651, 13488.2899],
    110: [-5.11921536e-06, 0.0752900004, 13363.3496],
    111: [-5.07017706e-06, 0.0746297729, 13240.6847],
    112: [-5.01192617e-06, 0.0739609738, 13120.2183],
    113: [-4.94013215e-06, 0.0732732739, 13001.8676],
    114: [-4.84942915e-06, 0.0725537869, 12885.5405],
    115: [-4.73328709e-06, 0.0717867142, 12771.1326],
    116: [-4.5838782e-06, 0.0709529713, 12658.5223],
    117: [-4.39193925e-06, 0.0700297962, 12547.566],
    118: [-4.14662982e-06, 0.0689903401, 12438.0922],
    119: [-3.83538672e-06, 0.0678032403, 12329.8942],
    120: [-3.44377496e-06, 0.0664321764, 12222.7231],
    121: [-2.95533525e-06, 0.0648354095, 12116.2779],
    122: [-2.35142842e-06, 0.0629653049, 12010.1966],
    123: [-1.61107682e-06, 0.0607678378, 11904.0438],
    124: [-7.10802964e-07, 0.0581820837, 11797.2991],
    125: [3.75534581e-07, 0.0551396921, 11689.3427],
    126: [1.6769077e-06, 0.0515643447, 11579.44],
    127: [3.22528768e-06, 0.0473711982, 11466.725],
    128: [5.05581638e-06, 0.0424663111, 11350.181],
    129: [7.20697948e-06, 0.0367460556, 11228.6214]
}