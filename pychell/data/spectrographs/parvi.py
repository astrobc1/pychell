# Base Python
import os
import copy
import glob
import pickle

# Astropy
from astropy.io import fits
import astropy.coordinates
import astropy.time
import astropy.units as units

# Maths
import numpy as np
import scipy.constants as cs

# Pychell deps
import pychell.data as pcdata
import pychell.maths as pcmath
import pychell.spectralmodeling.barycenter
from pychell.reduce.recipe import ReduceRecipe
import pychell.utils as pcutils
import pychell.reduce.precalib as pccalib
import pychell.spectralmodeling.combs as pccombs

# Site
observatory = "palomar"

# Orders
echelle_orders = [84, 129]

# Detector
detector = {"dark_current": 0.0, "gain": 1, "read_noise": 0}

# fwhm
lsf_sigma = [0.0095, 0.0095, 0.0095]

# Information to generate a crude ishell wavelength solution for the above method estimate_wavelength_solution
wls_pixel_lagrange_points = [199, 1023.5, 1847]

# LFC info
192.1852839999997
lfc_f0 = cs.c / (1559.91370 * 1E-9) # freq of pump line in Hz.
lfc_df = 10.0000000 * 1E9 # spacing of peaks [Hz]

# Useful area on detector
poly_bottom = np.array([-6.517857142857155e-05, 0.15125000000000033, -46.07142857142874])
poly_top = np.array([-6.041666666666651e-05, 0.12291666666666691, 1975.4999999999993])


# When orientated with orders along detector rows and concave down (vertex at top)
# Top fiber is 1 (sci)
# Bottom fiber is 3 (cal)
# Approximate quadratic length solution coeffs as a starting point
wls_coeffs_fiber1 = {84: [-6.53864897e-07, 0.00980298331, 1759.3386699999999], 85: [-6.46148117e-07, 0.009687404909999999, 1738.1397000000002], 86: [-6.38707844e-07, 0.00957457761, 1717.4456300000002], 87: [-6.31519485e-07, 0.00946445759, 1697.2385], 88: [-6.2456208e-07, 0.00935698607, 1677.5012599999998], 89: [-6.17817854e-07, 0.00925209401, 1658.21769], 90: [-6.11271815e-07, 0.00914970599, 1639.37234], 91: [-6.04911393e-07, 0.00904974352, 1620.95044], 92: [-5.98726128e-07, 0.0089521276, 1602.93786], 93: [-5.927073890000001e-07, 0.00885678082, 1585.32111], 94: [-5.86848138e-07, 0.00876362894, 1568.08726], 95: [-5.81142717e-07, 0.00867260208, 1551.22395], 96: [-5.75586675e-07, 0.0085836355, 1534.71932], 97: [-5.70176616e-07, 0.008496670080000001, 1518.56205], 98: [-5.6491008e-07, 0.00841165252, 1502.74127], 99: [-5.597854419999999e-07, 0.0083285353, 1487.24659], 100: [-5.548018420000001e-07, 0.00824727647, 1472.06801], 101: [-5.49959126e-07, 0.00816783919, 1457.19597], 102: [-5.45257819e-07, 0.0080901913, 1442.6212799999998], 103: [-5.40699106e-07, 0.00801430459, 1428.33509], 104: [-5.36284833e-07, 0.0079401542, 1414.32889], 105: [-5.32017532e-07, 0.00786771778, 1400.5945], 106: [-5.27900446e-07, 0.007796974789999999, 1387.1240400000002], 107: [-5.239375849999999e-07, 0.007727905679999999, 1373.90993], 108: [-5.20133781e-07, 0.00766049115, 1360.94491], 109: [-5.16494762e-07, 0.0075947114400000005, 1348.2220499999999], 110: [-5.13027239e-07, 0.0075305456500000005, 1335.73479], 111: [-5.09738999e-07, 0.00746797114, 1323.47699], 112: [-5.06639013e-07, 0.00740696303, 1311.44298], 113: [-5.03737548e-07, 0.0073474937700000005, 1299.62767], 114: [-5.010462919999999e-07, 0.00728953285, 1288.02668], 115: [-4.98578489e-07, 0.00723304661, 1276.63645], 116: [-4.963490739999999e-07, 0.00717799821, 1265.45444], 117: [-4.94374825e-07, 0.007124347709999999, 1254.47935], 118: [-4.92674517e-07, 0.007072052299999999, 1243.7113100000001], 119: [-4.9126908e-07, 0.00702106679, 1233.15224], 120: [-4.901817699999999e-07, 0.00697134412, 1222.80613], 121: [-4.894383380000001e-07, 0.00692283618, 1212.67945], 122: [-4.89067214e-07, 0.0068754947, 1202.78154], 123: [-4.890996849999999e-07, 0.0068292725, 1193.12513], 124: [-4.895700870000001e-07, 0.0067841247300000004, 1183.72687], 125: [-4.90515998e-07, 0.00674001048, 1174.6079100000002], 126: [-4.91978434e-07, 0.0066968945500000005, 1165.79459], 127: [-4.940020500000001e-07, 0.006654749409999999, 1157.31918], 128: [-4.96635349e-07, 0.00661355741, 1149.22064], 129: [-4.99930888e-07, 0.006573313209999999, 1141.5455900000002]}

wls_coeffs_fiber3 = {84: [-6.51759531e-07, 0.009800088580000001, 1760.1343499999998], 85: [-6.45430063e-07, 0.00968709018, 1738.92538], 86: [-6.386198559999999e-07, 0.00957541455, 1718.2215], 87: [-6.31621058e-07, 0.00946560766, 1698.0049800000002], 88: [-6.24631195e-07, 0.00935802387, 1678.25877], 89: [-6.17773991e-07, 0.00925286629, 1658.96656], 90: [-6.11117352e-07, 0.0091502223, 1640.11279], 91: [-6.046887470000001e-07, 0.00905009435, 1621.68261], 92: [-5.98488157e-07, 0.00895242649, 1603.6618700000001], 93: [-5.924988049999999e-07, 0.008857126810000001, 1586.0370699999999], 94: [-5.86695809e-07, 0.00876408587, 1568.79531], 95: [-5.81052925e-07, 0.00867319152, 1551.92427], 96: [-5.75547508e-07, 0.00858434015, 1535.4121300000002], 97: [-5.701638329999999e-07, 0.00849744455, 1519.24756], 98: [-5.64894883e-07, 0.008412438599999999, 1503.4196900000002], 99: [-5.597427150000001e-07, 0.00832927882, 1487.91809], 100: [-5.54717499e-07, 0.00824794307, 1472.73271], 101: [-5.49835342e-07, 0.0081684263, 1457.8539500000002], 102: [-5.45114946e-07, 0.00809073365, 1443.2725500000001], 103: [-5.40573212e-07, 0.008014870890000001, 1428.97968], 104: [-5.36219842e-07, 0.00794083236, 1414.96685], 105: [-5.32051014e-07, 0.00786858642, 1401.22593], 106: [-5.28042184e-07, 0.00779805856, 1387.74911], 107: [-5.24140082e-07, 0.00772911219, 1374.52888], 108: [-5.20253946e-07, 0.0076615272299999995, 1361.5579], 109: [-5.16246045e-07, 0.0075949765099999995, 1348.82899], 110: [-5.11921536e-07, 0.00752900004, 1336.33496], 111: [-5.07017706e-07, 0.0074629772900000006, 1324.06847], 112: [-5.01192617e-07, 0.007396097379999999, 1312.0218300000001], 113: [-4.94013215e-07, 0.007327327390000001, 1300.18676], 114: [-4.849429149999999e-07, 0.00725537869, 1288.55405], 115: [-4.7332870899999994e-07, 0.00717867142, 1277.11326], 116: [-4.5838782e-07, 0.00709529713, 1265.85223], 117: [-4.39193925e-07, 0.00700297962, 1254.7566000000002], 118: [-4.1466298200000005e-07, 0.006899034010000001, 1243.8092199999999], 119: [-3.83538672e-07, 0.00678032403, 1232.98942], 120: [-3.4437749600000004e-07, 0.00664321764, 1222.2723099999998], 121: [-2.9553352500000003e-07, 0.00648354095, 1211.62779], 122: [-2.3514284199999999e-07, 0.0062965304900000005, 1201.01966], 123: [-1.61107682e-07, 0.00607678378, 1190.40438], 124: [-7.10802964e-08, 0.0058182083699999994, 1179.72991], 125: [3.75534581e-08, 0.00551396921, 1168.93427], 126: [1.6769077e-07, 0.00515643447, 1157.944], 127: [3.22528768e-07, 0.00473711982, 1146.6725000000001], 128: [5.05581638e-07, 0.00424663111, 1135.0181], 129: [7.20697948e-07, 0.00367460556, 1122.86214]}

######################
#### DATA PARSING ####
######################

def parse_utdate(data):
    jd_start = astropy.time.Time(parse_exposure_start_time(data), scale='utc', format='jd')
    y, m, d = str(jd_start.datetime.year), str(jd_start.datetime.month), str(jd_start.datetime.day)
    if len(m) == 1:
        m = f"0{m}"
    if len(d) == 1:
        d = f"0{d}"
    utdate = "".join([str(y), str(m), str(d)])
    return utdate

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

def parse_image(data, scale_to_itime=True):
    image = fits.open(data.input_file, do_not_scale_image_data=True)[0].data.astype(float)
    if "master" in data.base_input_file:
        return image
    
    # Transpose
    image = image.T

    # Scale slope to itime
    #if scale_to_itime:
        #image *= parse_itime(data)
    
    return image

def parse_header(input_file):
    return fits.open(input_file)[0].header

def parse_object(data):
    return data.header["OBJECT"]
    
def parse_sky_coord(data):
    if data.header['P200RA'] is not None and data.header['P200DEC'] is not None:
        coord = astropy.coordinates.SkyCoord(ra=data.header['P200RA'], dec=data.header['P200DEC'], unit=(units.hourangle, units.deg))
    elif data.header['RA'] is not None and data.header['DEC'] is not None:
        coord = astropy.coordinates.SkyCoord(ra=data.header['RA'], dec=data.header['DEC'], unit=(units.hourangle, units.deg))
    else:
        coord = astropy.coordinates.SkyCoord(ra=np.nan, dec=np.nan, unit=(units.hourangle, units.deg))
    return coord

def parse_itime(data):
    return data.header["EXPTIME"]
    
def parse_exposure_start_time(data):
    return astropy.time.Time(float(data.header["TIMEI00"]) / 1E9, format="unix").jd

def parse_fiber_nums(data):
    #if "fiberflat" in data.base_input_file.lower() or "fibreflat" in data.base_input_file.lower():
    #    return [str(data.header["FIBER"])]
    #else:
    #    return ["1", "3"]
    return [str(data.header["FIBER"])]





def parse_spec1d(input_file, sregion):
    f = fits.open(input_file)
    f.verify('fix')
    oi = sregion.order - echelle_orders[0]
    #wave = estimate_order_wls(sregion.order, 1)
    #wave = pcmath.doppler_shift_wave(wave, f[0].header["lfcdrift"])
    wave = f[0].data[oi, :, 3]
    flux = f[0].data[oi, :, 0]
    #flux = f[0].data[oi, :, 4]
    fluxerr = f[0].data[oi, :, 1]
    mask = f[0].data[oi, :, 2]
    medval = pcmath.weighted_median(flux, percentile=0.99)
    flux /= medval
    fluxerr /= medval
    wave[0:sregion.pixmin] = np.nan
    wave[sregion.pixmax:] = np.nan
    flux[0:sregion.pixmin] = np.nan
    flux[sregion.pixmax:] = np.nan
    mask[0:sregion.pixmin] = 0
    mask[sregion.pixmax:] = 0
    data = {"wave": wave, "flux": flux, "fluxerr": fluxerr, "mask": mask}
    return data

#######################
#### GROUPING CALS ####
#######################

def group_darks(darks):
    return [darks]

def group_flats(flats):
    return [flats]

def gen_master_calib_filename(master_cal):
    fname0 = master_cal.group[0].base_input_file.lower()
    utdate = parse_utdate(master_cal.group[0])
    if "dark" in fname0:
        itime = parse_itime(master_cal.group[0])
        return f"master_dark_{itime}s.fits"
    elif "fiberflat" in fname0 or "fibreflat" in fname0:
        return f"master_fiberflat_{utdate}_fiber{parse_fiber_nums(master_cal.group[0])[0]}.fits"
    elif "fullflat" in fname0:
        return f"master_fullflat_{utdate}.fits"
    elif "lfc" in fname0:
        return f"master_lfc_{utdate}_fiber{parse_fiber_nums(master_cal.group[0])[0]}.fits"
    else:
        return f"master_calib_{utdate}.fits"

def gen_master_calib_header(master_cal):
    return copy.deepcopy(master_cal.group[0].header)


################################
#### BARYCENTER CORRECTIONS ####
################################

def get_barycenter_corrections(data, star_name):
    jdmid = get_exposure_midpoint(data)
    bjd, bc_vel = pychell.spectralmodeling.barycenter.compute_barycenter_corrections(jdmid, star_name, observatory)
    return bjd, bc_vel

def get_exposure_midpoint(data):
    #jdsi, jdsf = [], []
    # for key in data.header:
    #     if key.startswith("TIMEI"):
    #         jdsi.append(astropy.time.Time(float(data.header[key]) / 1E9, format="unix").jd)
    #     if key.startswith("TIMEF"):
    #         jdsf.append(astropy.time.Time(float(data.header[key]) / 1E9, format="unix").jd)
    # jdmid = (np.nanmax(jdsf) - np.nanmin(jdsi)) / 2 + np.nanmin(jdsi)
    jdmid = parse_exposure_start_time(data) + parse_itime(data) / (2 * 86400)
    return jdmid

###################
#### WAVE INFO ####
###################

def estimate_wls(data, sregion):
    if sregion.fiber == 1:
        pcoeffs = wls_coeffs_fiber1[sregion.order]
    else:
        pcoeffs = wls_coeffs_fiber3[sregion.order]
    wls = np.polyval(pcoeffs, np.arange(2048).astype(float))
    return wls


def estimate_order_wls(order, fiber=1):
    if fiber == 1:
        pcoeffs = wls_coeffs_fiber1[order]
    else:
        pcoeffs = wls_coeffs_fiber3[order]
    wls = np.polyval(pcoeffs, np.arange(2048) - 65)
    return wls


################################
#### REDUCTION / EXTRACTION ####
################################

ccpoly_coeffs_sigma_fiber1 = np.array([96073.77705017618, -159364.74850970478, 89169.12439151623, -30943.450178612766, 5262.963082028526, -147877.14310955434, 245260.7770622876, -137152.9813391628, 47547.111356818714, -8070.310077940552, 79775.40550393143, -132376.99027062586, 74151.33624405166, -25781.907133836587, 4400.228885654104])

ccpoly_coeffs_sigma_fiber3 = np.array([94049.5228651916, -156002.95416474718, 87280.18566312999, -30283.15892820581, 5148.960163410947, -144807.18259649677, 240162.83920815645, -134289.53099277674, 46546.7647985848, -7897.784267636158, 78899.84589777053, -130922.2203709168, 73332.61292589683, -25494.939528154355, 4350.430335676172])

ccpoly_coeffs_q_fiber1 = np.array([232.93922991026938, -354.9856514408896, 150.26844346683168, -25.956251261522915, -395.2583109695755, 607.7341170788181, -261.0387359006901, 47.28075765353607, 427.9161319765941, -669.218730381846, 302.5680778372938, -63.46430238829523])

ccpoly_coeffs_q_fiber3 = np.array([191.75618488457476, -290.04886517645383, 120.24041870599879, -19.26114678052877, -341.6917236682601, 523.2717695034502, -221.98241622951923, 38.57307711434833, 395.441504715445, -618.0128368237691, 278.88902247804083, -58.184520689727655])


theta_fiber1 = np.pi/2 + np.pi/4
theta_fiber3 = np.pi/2 + np.pi/4

def get_psf_parameters(data, order, fiber=1):
    nx = 2048
    xarr = np.arange(nx)
    max_order = echelle_orders[1]
    if fiber == 1:
        sigmas = pcmath.chebyval2d2(ccpoly_coeffs_sigma_fiber1, xarr / nx, np.full(nx, order / max_order), poly_order_inter_order=2, poly_order_intra_order=4)
        qs = pcmath.chebyval2d2(ccpoly_coeffs_q_fiber1, xarr / nx, np.full(nx, order / max_order), poly_order_inter_order=2, poly_order_intra_order=3)
        thetas = np.full(nx, theta_fiber1)
    else:
        sigmas = pcmath.chebyval2d2(ccpoly_coeffs_sigma_fiber3, xarr / nx, np.full(nx, order / max_order), poly_order_inter_order=2, poly_order_intra_order=4)
        qs = pcmath.chebyval2d2(ccpoly_coeffs_q_fiber3, xarr / nx, np.full(nx, order / max_order), poly_order_inter_order=2, poly_order_intra_order=3)
        thetas = np.full(nx, theta_fiber3)

    return sigmas, qs, thetas




class PARVIReduceRecipe(ReduceRecipe):
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, target_input_paths, utdate, output_path,
                 full_flat_files=None, fiber_flat_files=None, dark_files=None, lfc_zero_point_files=None, badpix_mask_file=None,
                 do_bias=False, do_dark=True, do_flat=True,
                 sregion=None, tracer=None, extractor=None,
                 n_cores=1):

        # The spectrograph
        self.spectrograph = "PARVI"

        # Store additional PARVI related params
        self.target_input_paths = target_input_paths
        self.full_flat_files = full_flat_files
        self.fiber_flat_files = fiber_flat_files
        self.dark_files = dark_files
        self.lfc_zero_point_files = lfc_zero_point_files
        self.utdate = utdate
        self.badpix_mask_file = badpix_mask_file
        
        # Number of cores
        self.n_cores = n_cores
        
        # The output path
        self.target_output_paths = {t.split(os.sep)[-2]: output_path + t.split(os.sep)[-2] + os.sep for t in target_input_paths}
        self.calib_output_path = output_path + "calib_" + self.utdate + os.sep

        # Pre calibration
        self.do_bias = do_bias
        self.do_flat = do_flat
        self.do_dark = do_dark

        # Image area
        self.sregion = sregion

        # Reduction steps
        self.tracer = tracer
        self.extractor = extractor

        # Init the data
        self.init_data()

    def init_data(self):
        """Initialize the data by calling self.categorize_raw_data.
        """
        
        # Identify what's what.
        print("Categorizing Data ...", flush=True)
        self.data = self.categorize_raw_data()

        # Print reduction summary
        self.print_reduction_summary()


    def create_output_dirs(self):
        """Creates the output folder and subfolders trace, spectra, calib.
        """
    
        # Make the root output directory for this run
        for t in self.target_output_paths.values():
            os.makedirs(t, exist_ok=True)

        # Trace information (positions, profiles)
        os.makedirs(self.calib_output_path, exist_ok=True)

    def reduce(self):
        """Primary method to reduce a given directory.
        """

        # Start the main clock
        stopwatch = pcutils.StopWatch()

        # Create the output directories
        self.create_output_dirs()

        # Generate pre calibration images
        self.gen_master_calib_images()
        
        # Trace orders for appropriate images
        self.trace()
        
        # Extract all desired images
        self.extract()
        
        # Run Time
        print(f"REDUCTION COMPLETE! TOTAL TIME: {round(stopwatch.time_since() / 3600, 2)} hours")

    def trace(self):
        """Traces the orders.
        """
        order_map = self.data['master_fiber_flat_fiber1']
        print(f"Tracing orders for {order_map} ...", flush=True)
        self.tracer.trace(order_map, self.sregion, fiber=1)
        with open(f"{self.calib_output_path}{order_map.base_input_file_noext}_order_map.pkl", 'wb') as f:
            pickle.dump(order_map.orders_list, f)

        order_map = self.data['master_fiber_flat_fiber3']
        print(f"Tracing orders for {order_map} ...", flush=True)
        self.tracer.trace(order_map, self.sregion, fiber=3)
        with open(f"{self.calib_output_path}{order_map.base_input_file_noext}_order_map.pkl", 'wb') as f:
            pickle.dump(order_map.orders_list, f)

    def gen_master_calib_images(self):
        if self.do_dark:
            master_dark = pccalib.gen_master_dark(self.data['master_dark'], self.do_bias)
            self.data['master_dark'].save(master_dark)
        if self.do_flat:
            master_full_flat = pccalib.gen_master_flat(self.data['master_flat'], self.do_bias, self.do_dark)
            self.data['master_flat'].save(master_full_flat)

        image = pccalib.gen_coadded_master_image(self.data['master_fiber_flat_fiber1'])
        self.data['master_fiber_flat_fiber1'].save(image)

        image = pccalib.gen_coadded_master_image(self.data['master_fiber_flat_fiber3'])
        self.data['master_fiber_flat_fiber3'].save(image)

        image = pccalib.gen_coadded_master_image(self.data['master_lfc_fiber1'])
        self.data['master_lfc_fiber1'].save(image)

        image = pccalib.gen_coadded_master_image(self.data['master_lfc_fiber3'])
        self.data['master_lfc_fiber3'].save(image)


    @staticmethod
    def extract_image_wrapper(recipe, data):
        """Wrapper to extract an image for parallel processing. Performs the pre calibration.

        Args:
            recipe (ReduceRecipe): The recpe object.
            data (RawEchellogram): The data object.
        """

        # Stopwatch
        stopwatch = pcutils.StopWatch()

        # Print start
        print(f"Extracting {data} ...")

        # Load image
        data_image = data.parse_image()

        # Calibrate image
        pccalib.pre_calibrate(data, data_image, recipe.do_bias, recipe.do_dark, recipe.do_flat)

        # Mask
        if 'badpix_mask' in recipe.data:
            badpix_mask = recipe.data['badpix_mask']
        else:
            badpix_mask = None
        
        # Extract image
        if data in recipe.data['science']:
            output_path = recipe.target_output_paths[parse_object(data)]
        else:
            output_path = recipe.calib_output_path
        recipe.extractor.extract_image(data, data_image, recipe.sregion, output_path, badpix_mask=badpix_mask)
        
        # Print end
        print(f"Extracted {data} in {round(stopwatch.time_since() / 60, 2)} min")
        

    def get_wavelength_solutions1d(self, poly_order=6, use_orders=None):

        nx = 2048
        n_orders = self.sregion.n_orders
        n_sci = len(self.data['science'])

        # Wave estimates for each order and fiber
        wave_estimates_fiber1 = np.full((n_orders, nx), np.nan)
        wave_estimates_fiber3 = np.full((n_orders, nx), np.nan)
        for o in range(n_orders):
            order = self.sregion.ordermin + o
            wave_estimates_fiber1[o, :] = estimate_order_wls(order, fiber=1)
            wave_estimates_fiber3[o, :] = estimate_order_wls(order, fiber=3)

        # Echelle orders
        echelle_orders = np.arange(self.sregion.ordermin, self.sregion.ordermax + 1)

        # Parse LFC flux results
        fname_lfc_fiber1 = glob.glob(f"{self.calib_output_path}*master_lfc*fiber1*reduced.fits")[0]
        fname_lfc_fiber3 = glob.glob(f"{self.calib_output_path}*master_lfc*fiber3*reduced.fits")[0]
        lfc_flux_fiber1 = fits.open(fname_lfc_fiber1)[0].data[:, 0, :, 0]
        lfc_flux_fiber3 = fits.open(fname_lfc_fiber3)[0].data[:, 0, :, 0]
        import matplotlib.pyplot as plt; import matplotlib; matplotlib.use("MacOSX")
        pcoeffs0_fiber1 = {}
        pcoeffs0_fiber3 = {}
        for o in range(len(echelle_orders)):
            order = echelle_orders[o]
            if order in use_orders:
                result = pccombs.compute_wls1d(wave_estimates_fiber1[o, :], lfc_flux_fiber1[o, :], lfc_f0, lfc_df, poly_order=poly_order)
                pcoeffs0_fiber1[order] = result[1]
                result = pccombs.compute_wls1d(wave_estimates_fiber3[o, :], lfc_flux_fiber3[o, :], lfc_f0, lfc_df, poly_order=poly_order)
                pcoeffs0_fiber3[order] = result[1]

        drifts = np.full((n_sci, n_orders), np.nan)
        stddevs = np.full((n_sci, n_orders), np.nan)
        wls_fiber1 = np.full((len(self.data['science']), n_orders, nx), np.nan)
        for i, sci in enumerate(self.data['science']):
            target = parse_object(sci)
            fname = glob.glob(f"{self.target_output_paths[target]}{sci.base_input_file_noext}*reduced.fits")[0]
            lfc_fluxes = fits.open(fname)[0].data[:, 1, :, 0]
            for o in range(len(echelle_orders)):
                order = echelle_orders[o]
                if order in use_orders:
                    #breakpoint() # import matplotlib; import matplotlib.pyplot as plt; matplotlib.use("MacOSX");
                    # f1 = lfc_flux_fiber3[o, :]
                    # b1 = pcmath.cspline_fit_fancy(np.arange(nx), f1, window=15, n_knots=100, percentile=0.0001)
                    # c1 = pcmath.cspline_fit_fancy(np.arange(nx), f1 - b1, window=15, n_knots=100, percentile=0.99999)
                    # f2 = lfc_fluxes[o, :]
                    # b2 = pcmath.cspline_fit_fancy(np.arange(nx), f2, window=15, n_knots=100, percentile=0.0001)
                    # c2 = pcmath.cspline_fit_fancy(np.arange(nx), f2 - b2, window=15, n_knots=100, percentile=0.99999)
                    #plt.plot((f1 - b1) / c1); plt.plot((f2 - b2) / c2); plt.show()
                    pixel_centers, wave_centers, rms_norm, peak_integers = pccombs.compute_peaks(wave_estimates_fiber3[o, :], lfc_fluxes[o, :], lfc_f0, lfc_df, xrange=[100, 1948])
                    weights = 1 / rms_norm**2
                    #breakpoint()
                    #import matplotlib; import matplotlib.pyplot as plt; matplotlib.use("MacOSX"); plt.plot(lfc_centers_pix, pcmath.dl_to_dv(lfc_centers_wave - np.polyval(pfit, lfc_centers_pix), lfc_centers_wave), marker='o', lw=0); plt.ylabel("Residuals [m/s]"); plt.xlabel("Pixels"); plt.title("March 13, Cal Frame, Cal fiber, Order 96"); plt.ylim(-100, 100); plt.tight_layout(); plt.savefig("residuals2.png", dpi=200); plt.close()
                    #import matplotlib; import matplotlib.pyplot as plt; matplotlib.use("MacOSX"); plt.plot(pixel_centers, pcmath.dl_to_dv(wave_centers - np.polyval(pcoeffs0_fiber3[order], pixel_centers), wave_centers), marker='o', lw=0); plt.ylabel("Residuals [m/s]"); plt.xlabel("Pixels"); plt.title("March 13, Sci Frame, Cal fiber, Order 96"); plt.ylim(-100, 100); plt.tight_layout(); plt.savefig("residuals5.png", dpi=200); plt.close()
                    #breakpoint()
                    _drifts = wave_centers - np.polyval(pcoeffs0_fiber3[order], pixel_centers)
                    _drifts = pcmath.dl_to_dv(_drifts, wave_centers)
                    drifts[i, o] = pcmath.weighted_median(_drifts, weights)
                    use = np.where((pixel_centers > 300) & (pixel_centers < 2048-300))[0]
                    #drifts[i, o] = np.nanmedian(_drifts[use])
                    stddevs[i, o] = pcmath.weighted_stddev(_drifts[use], weights[use]) / np.sqrt(len(_drifts[use]))
                    print(order, drifts[i, o], stddevs[i, o])
                    #wls_fiber1[i, o, :] = pcmath.doppler_shift_wave(np.polyval(pcoeffs0_fiber1[order], np.arange(nx)), drifts[i, o])
                    wls_fiber1[i, o, :] = np.copy(np.polyval(pcoeffs0_fiber1[order], np.arange(nx)))
        
        return wls_fiber1, drifts, stddevs


    def get_wavelength_solutions2d(self, use_orders=None, poly_order_inter_order=4, poly_order_intra_order=4):

        nx = 2048
        n_orders = self.sregion.n_orders
        n_sci = len(self.data['science'])

        # Wave estimates for each order and fiber
        wave_estimates_fiber1 = np.full((n_orders, nx), np.nan)
        wave_estimates_fiber3 = np.full((n_orders, nx), np.nan)
        for o in range(n_orders):
            order = self.sregion.ordermin + o
            wave_estimates_fiber1[o, :] = estimate_order_wls(order, fiber=1)
            wave_estimates_fiber3[o, :] = estimate_order_wls(order, fiber=3)

        # Echelle orders
        echelle_orders = np.arange(self.sregion.ordermin, self.sregion.ordermax + 1)
        if use_orders is None:
            use_orders = echelle_orders
        max_order = np.max(echelle_orders)

        # Parse LFC flux results
        fname_lfc_fiber1 = glob.glob(f"{self.calib_output_path}*master_lfc*fiber1*reduced.fits")[0]
        fname_lfc_fiber3 = glob.glob(f"{self.calib_output_path}*master_lfc*fiber3*reduced.fits")[0]
        lfc_flux_fiber1 = fits.open(fname_lfc_fiber1)[0].data[:, 0, :, 0]
        lfc_flux_fiber3 = fits.open(fname_lfc_fiber3)[0].data[:, 0, :, 0]

        # Generate wls for zero point
        pcoeffs0_fiber1, wls2d0_fiber1, pixel_peaks0_fiber1, wave_peaks0_fiber1, rms0_fiber1, peak_integers0_fiber1 = pccombs.compute_chebyshev_wls_2d(lfc_f0, lfc_df, wave_estimates_fiber1, lfc_flux_fiber1, echelle_orders, use_orders, poly_order_inter_order, poly_order_intra_order)
        pcoeffs0_fiber3, wls2d0_fiber3, pixel_peaks0_fiber3, wave_peaks0_fiber3, rms0_fiber3, peak_integers0_fiber3 = pccombs.compute_chebyshev_wls_2d(lfc_f0, lfc_df, wave_estimates_fiber3, lfc_flux_fiber3, echelle_orders, use_orders, poly_order_inter_order, poly_order_intra_order)

        # Generate wls for science from drift
        #drifts = np.full((n_sci, n_orders), np.nan)
        #stddevs = np.full((n_sci, n_orders), np.nan)
        drifts = np.full(n_sci, np.nan)
        stddevs = np.full(n_sci, np.nan)
        wls_fiber1 = np.full((n_sci, n_orders, nx), np.nan)
        for i, sci in enumerate(self.data['science']):
            target = parse_object(sci)
            fname = glob.glob(f"{self.target_output_paths[target]}{sci.base_input_file_noext}*reduced.fits")[0]
            lfc_fluxes = fits.open(fname)[0].data[:, 1, :, 0]


        #     for o in range(len(echelle_orders)):
        #         order = echelle_orders[o]
        #         if order in use_orders:
        #             #pixel_centers, wave_centers, rms, integers = pccombs.compute_peaks(wave_estimates_fiber3[o, :], lfc_fluxes[o, :], lfc_f0, lfc_df, xrange=[150, 1900])
        #             #breakpoint()
        #             #echelle_orders_flat = np.full(len(pixel_centers), order)
        #             #wls2d0 = pccombs.build_chebyshev_wls_2d(pcoeffs0_fiber3, echelle_orders_flat, pixel_centers, nx, max_order, poly_order_inter_order, poly_order_intra_order)
        #             #weights = 1 / rms**2
        #             drifts[i, o], stddevs[i, o] = pccombs.compute_drift2d(lfc_f0, lfc_df, pcoeffs0_fiber3, lfc_fluxes, echelle_orders, [order], poly_order_inter_order, poly_order_intra_order)
        #             print(order, drifts[i, o], stddevs[i, o])
        #     wls_fiber1[i, :, :] = np.copy(wls2d0_fiber1)

            drifts[i], stddevs[i] = pccombs.compute_drift2d(lfc_f0, lfc_df, pcoeffs0_fiber3, lfc_fluxes, echelle_orders, use_orders, poly_order_inter_order, poly_order_intra_order)
            wls_fiber1[i, :, :] = np.copy(wls2d0_fiber1)

            print(f"ALL: {drifts[i], stddevs[i]}")

        
        # Return wls for all science observations
        return wls_fiber1, drifts, stddevs

    def get_continuum_corrections(self):
        nx = 2048
        fname_fiber_flat_fiber1 = glob.glob(f"{self.calib_output_path}*master_fiberflat*fiber1*reduced.fits")[0]
        fiber_flat_fiber1 = fits.open(fname_fiber_flat_fiber1)[0].data
        xarr = np.arange(nx)
        n_orders = self.sregion.n_orders
        continua = np.full((n_orders, nx), np.nan)
        for o in range(n_orders):
            order = self.sregion.ordermin + o
            if order in self.extractor.extract_orders:
                ff_smooth = pcmath.median_filter1d(fiber_flat_fiber1[o, 0, :, 0], width=31)
                continua[o, :] = pcmath.poly_filter(ff_smooth, width=501, poly_order=2)
        return continua


    def categorize_raw_data(self):

        data = {}

        # Classify files
        sci_files = []
        for t in self.target_input_paths:
            sci_files += glob.glob(t + f"*{self.utdate}*.fits")

        # Create Echellograms from raw data
        data['science'] = [pcdata.Echellogram(input_file=f, spectrograph="PARVI") for f in sci_files]
        data['fiber_flats_fiber1'] = [pcdata.Echellogram(input_file=f, spectrograph="PARVI") for f in self.fiber_flat_files[1]]
        data['fiber_flats_fiber3'] = [pcdata.Echellogram(input_file=f, spectrograph="PARVI") for f in self.fiber_flat_files[3]]
        data['darks'] = [pcdata.Echellogram(input_file=f, spectrograph="PARVI") for f in self.dark_files]
        data['full_flats'] = [pcdata.Echellogram(input_file=f, spectrograph="PARVI") for f in self.full_flat_files]
        if self.lfc_zero_point_files is not None:
            data['lfc_fiber1'] = [pcdata.Echellogram(input_file=f, spectrograph="PARVI") for f in self.lfc_zero_point_files[1]]
            data['lfc_fiber3'] = [pcdata.Echellogram(input_file=f, spectrograph="PARVI") for f in self.lfc_zero_point_files[3]]

        # Master Darks
        if len(self.dark_files) > 0:
            data['master_dark'] = pcdata.MasterCal(data['darks'], self.calib_output_path)

        # Master Flats
        if len(self.full_flat_files) > 0:
            data['master_flat'] = pcdata.MasterCal(data['full_flats'], self.calib_output_path)
        
        # Master fiber flats
        data['master_fiber_flat_fiber1'] = pcdata.MasterCal(data['fiber_flats_fiber1'], self.calib_output_path)
        data['master_fiber_flat_fiber3'] = pcdata.MasterCal(data['fiber_flats_fiber3'], self.calib_output_path)

        if self.lfc_zero_point_files is not None:
            data['master_lfc_fiber1'] = pcdata.MasterCal(data['lfc_fiber1'], self.calib_output_path)
            data['master_lfc_fiber3'] = pcdata.MasterCal(data['lfc_fiber3'], self.calib_output_path)

        # Pair cals for science
        for d in data['science']:
            d.order_maps = [data['master_fiber_flat_fiber1'], data['master_fiber_flat_fiber3']]
            d.master_dark = data['master_dark']
            d.master_flat = data['master_flat']

        # Pair cals for individual full frame flats
        d = data['master_flat']
        if len(self.dark_files) > 0:
            d.master_dark = data['master_dark']

        # Pair cals for individual fiber flats
        d = data['master_fiber_flat_fiber1']
        d.order_maps = [data['master_fiber_flat_fiber1']]
        d.master_flat = data['master_flat']
        if len(self.dark_files) > 0:
            d.master_dark = data['master_dark']

        d = data['master_fiber_flat_fiber3']
        d.order_maps = [data['master_fiber_flat_fiber3']]
        d.master_flat = data['master_flat']
        if len(self.dark_files) > 0:
            d.master_dark = data['master_dark']

        # Pair cals for individual lfc cals
        d = data['master_fiber_flat_fiber1']
        d.order_maps = [data['master_fiber_flat_fiber1']]
        d.master_flat = data['master_flat']
        if len(self.dark_files) > 0:
            d.master_dark = data['master_dark']

        # Pair cals for individual lfc cals
        if self.lfc_zero_point_files is not None:
            d = data['master_lfc_fiber1']
            d.order_maps = [data['master_fiber_flat_fiber1']]
            d.master_flat = data['master_flat']
            if len(self.dark_files) > 0:
                d.master_dark = data['master_dark']

            d = data['master_lfc_fiber3']
            d.order_maps = [data['master_fiber_flat_fiber3']]
            d.master_flat = data['master_flat']
            if len(self.dark_files) > 0:
                d.master_dark = data['master_dark']

        # Which to extract
        data['extract'] = data['science'] + [data['master_fiber_flat_fiber1']] + [data['master_fiber_flat_fiber3']] + [data['master_lfc_fiber1']] + [data['master_lfc_fiber3']]

        # Bad pixel mask (only one, load into memory)
        data['badpix_mask'] = 1 - fits.open(self.badpix_mask_file)[0].data.astype(float)

        return data
