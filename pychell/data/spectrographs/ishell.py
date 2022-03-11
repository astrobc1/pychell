# imports
import os
import copy
import glob
import pickle
from astropy.io import fits
import astropy.coordinates
import astropy.time
import astropy.units as units
import numpy as np
import sklearn.cluster
import pychell.data as pcdata
import pychell.maths as pcmath
import pychell.spectralmodeling.barycenter
from pychell.reduce.recipes import ReduceRecipe
import pychell.utils as pcutils
import pychell.reduce.precalib as pccalib

# Site
observatory = "irtf"

# Orders
echelle_orders = [212, 240]

# Detector
detector = {"dark_current": 0.05, "gain": 1.8, "read_noise": 8.0}

# Gas cell
gascell_depth = [0.97, 0.97, 0.97]
gascell_file = "methane_gas_cell_ishell_kgas.npz"

# fwhm
lsf_sigma = [0.008 , 0.011, 0.015]

# Information to generate a crude ishell wavelength solution for the above method estimate_wavelength_solution
wls_pixel_lagrange_points = [199, 1023.5, 1847]

# Approximate for Kgas, correspond to wls_pixel_lagrange_points for each order in kgas
wls_wave_lagrange_points = {

    # Absolute echelle orders
    212: [2454.557561435, 2462.837672608, 2470.5724728630003],
    213: [2443.148444449, 2451.379686837, 2459.091231465],
    214: [2431.8408307639997, 2440.032734124, 2447.699298677],
    215: [2420.635776048, 2428.785495107, 2436.412010878],
    216: [2409.533986576, 2417.64424356, 2425.231443701],
    217: [2398.537381209, 2406.6078806220003, 2414.1555270910003],
    218: [2387.643046386, 2395.67243081, 2403.182506843],
    219: [2376.848974584, 2384.839610577, 2392.3122912139997],
    220: [2366.154443537, 2374.105658955, 2381.540789995],
    221: [2355.5563592089998, 2363.468688897, 2370.8701069070003],
    222: [2345.0551363570003, 2352.929771645, 2360.295596074],
    223: [2334.64923953, 2342.486836784, 2349.8186079409998],
    224: [2324.338904298, 2332.1379387, 2339.435163611],
    225: [2314.119183839, 2321.880573474, 2329.144815827],
    226: [2303.9902726249998, 2311.71876433, 2318.949231662],
    227: [2293.950127095, 2301.64487031, 2308.842080084],
    228: [2284.0009072420003, 2291.661245655, 2298.826540094],
    229: [2274.1403442250003, 2281.7657688890004, 2288.897654584],
    230: [2264.36481698, 2271.9564668020003, 2279.057559244],
    231: [2254.674892171, 2262.234315996, 2269.302942496],
    232: [2245.0709341770003, 2252.596723597, 2259.633915038],
    233: [2235.549187891, 2243.041612825, 2250.049456757],
    234: [2226.108953053, 2233.571472399, 2240.549547495],
    235: [2216.742305394, 2224.183394135, 2231.1255745589997],
    236: [2207.4728481360003, 2214.8736803809998, 2221.791297633],
    237: [2198.275611957, 2205.642903627, 2212.533774808],
    238: [2189.149178289, 2196.491093944, 2203.3503565250003],
    239: [2180.107332421, 2187.420764171, 2194.241058186],
    240: [2171.143496504, 2178.420091295, 2185.2242535550004]
}

######################
#### DATA PARSING ####
######################

def parse_header(input_file):
    return fits.open(input_file)[0].header

def parse_itime(data):
    return data.header["ITIME"]

def parse_object(data):
    return data.header["OBJECT"]

def parse_utdate(data):
    return "".join(data.header["DATE_OBS"].split('-'))

def parse_sky_coord(data):
    coord = astropy.coordinates.SkyCoord(ra=data.header['TCS_RA'], dec=data.header['TCS_DEC'], unit=(units.hourangle, units.deg))
    return coord
    
def parse_exposure_start_time(data):
    return astropy.time.Time(float(data.header['TCS_UTC']) + 2400000.5, scale='utc', format='jd').jd

def parse_image(data):
    image = fits.open(data.input_file, do_not_scale_image_data=True)[0].data.astype(float)
    correct_readmath(data, image)
    return image

def parse_spec1d(input_file, sregion):
    
    # Extract and flip
    oi = sregion.order - echelle_orders[0]
    f = fits.open(input_file)
    flux = f[0].data[oi, 0, ::-1, 0].astype(float)
    fluxerr = f[0].data[oi, 0, ::-1, 1].astype(float)
    mask = f[0].data[oi, 0, ::-1, 2].astype(float)
    flux = flux[sregion.pixmin:sregion.pixmax+1]
    fluxerr = fluxerr[sregion.pixmin:sregion.pixmax+1]
    mask = mask[sregion.pixmin:sregion.pixmax+1]
    medval = pcmath.weighted_median(flux, percentile=0.99)
    flux = flux / medval
    fluxerr = fluxerr / medval
    data = {"flux": flux,
            "fluxerr": fluxerr,
            "mask": mask}
    return data


###################
#### REDUCTION ####
###################


def categorize_raw_data(data_input_path, output_path):

    # Stores the data as above objects
    data = {}
    
    # iSHELL science files are files that contain spc or data
    sci_files = glob.glob(data_input_path + "*data*.fits") + glob.glob(data_input_path + "*spc*.fits")
    sci_files = np.sort(np.unique(np.array(sci_files, dtype='<U200'))).tolist()
    data['science'] = [pcdata.Echellogram(input_file=sci_file, spectrograph="iSHELL") for sci_file in sci_files]

    # Delete bad objects
    target_names = np.array([parse_object(d).lower() for d in data['science']], dtype='<U100')
    bad = np.where(target_names == "dark")[0]
    for i in sorted(bad, reverse=True):
        del data['science'][i]
    
    # iSHELL flats must contain flat in the filename
    flat_files = glob.glob(data_input_path + '*flat*.fits')
    if len(flat_files) > 0:
        data['flats'] = [pcdata.Echellogram(input_file=flat_files[f], spectrograph="iSHELL") for f in range(len(flat_files))]
        flat_groups = group_flats(data['flats'])
        data['master_flats'] = [pcdata.MasterCal(flat_group, output_path + "calib" + os.sep) for flat_group in flat_groups]
    
        for sci in data['science']:
            pair_master_flat(sci, data['master_flats'])
        
        # Order maps for iSHELL are the flat fields closest in time and space (RA+Dec) to the science target
        data['order_maps'] = data['master_flats']
        for sci_data in data['science']:
            pair_order_maps(sci_data, data['order_maps'])

    # Which to extract
    data['extract'] = data['science']

    # Return the data dict
    return data

def pair_order_maps(data, order_maps):
    for order_map in order_maps:
        if order_map == data.master_flat:
            data.order_maps = [order_map]

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

def gen_master_calib_filename(master_cal):
    fname0 = master_cal.group[0].base_input_file.lower()
    if "dark" in fname0:
        return f"master_dark_{parse_utdate(master_cal.group[0])}_{parse_itime(master_cal.group[0])}s.fits"
    elif "flat" in fname0:
        img_nums = np.array([parse_image_num(d) for d in master_cal.group], dtype=int)
        img_start, img_end = img_nums.min(), img_nums.max()
        return f"master_flat_{parse_utdate(master_cal.group[0])}_imgs{img_start}-{img_end}.fits"
    else:
        return f"master_calib.fits"

def gen_master_calib_header(master_cal):
    return copy.deepcopy(master_cal.group[0].header)

def parse_image_num(data):
    return data.base_input_file.split('.')[4]

def pair_master_bias(data, master_bias):
    data.master_bias = master_bias

def pair_master_dark(data, master_darks):
    itimes = np.array([master_darks[i].itime for i in range(len(master_darks))], dtype=float)
    good = np.where(data.itime == itimes)[0]
    if good.size != 1:
        raise ValueError(str(good.size) + " master dark(s) found for\n" + str(data))
    else:
        data.master_dark = master_darks[good[0]]

def pair_master_flat(data, master_flats):
    ang_seps = np.array([np.abs(parse_sky_coord(master_flat.group[0]).separation(parse_sky_coord(data)).value) for master_flat in master_flats], dtype=float)
    ang_seps /= 90
    time_seps = np.array([np.abs(parse_exposure_start_time(master_flat.group[0]) - parse_exposure_start_time(data)) for master_flat in master_flats], dtype=float)
    time_seps /= 100
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
            coordi = parse_sky_coord(flats[i])
            coordj = parse_sky_coord(flats[j])
            dpsi = coordi.separation(coordj).value
            dt = np.abs(parse_exposure_start_time(flats[i]) - parse_exposure_start_time(flats[j]))
            dt /= 100
            dpsi /= 90
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

################################
#### BARYCENTER CORRECTIONS ####
################################

def get_exposure_midpoint(data):
    return parse_exposure_start_time(data) + parse_itime(data) / (2 * 86400)

def get_barycenter_corrections(data, star_name):
    jdmid = get_exposure_midpoint(data)
    bjd, bc_vel = pychell.spectralmodeling.barycenter.compute_barycenter_corrections(jdmid, star_name, observatory)
    return bjd, bc_vel

#########################
#### BASIC WAVE INFO ####
#########################

def estimate_wls(data, sregion):
    wls = estimate_order_wls(sregion.order)
    wls = wls[sregion.pixmin:sregion.pixmax+1]
    return wls

def estimate_order_wls(order):
    pfit = np.polyfit(wls_pixel_lagrange_points, wls_wave_lagrange_points[order], 2)
    wls = np.polyval(pfit, np.arange(2048))
    return wls


################################
#### REDUCTION / EXTRACTION ####
################################

class iSHELLReduceRecipe(ReduceRecipe):

    def __init__(self, data_input_path, output_path, base_flat_field_file, do_bias=False, do_dark=True, do_flat=True, flat_percentile=0.5, xrange=None, poly_mask_bottom=None, poly_mask_top=None, tracer=None, extractor=None, n_cores=1):
        super().__init__(spectrograph="iSHELL",
                         data_input_path=data_input_path, output_path=output_path,
                         do_bias=do_bias, do_dark=do_dark, do_flat=do_flat, flat_percentile=flat_percentile,
                         xrange=xrange, poly_mask_top=poly_mask_top, poly_mask_bottom=poly_mask_bottom,
                         tracer=tracer, extractor=extractor, n_cores=n_cores)
        self.base_flat_field_file = base_flat_field_file

    def trace(self):
        """Traces the orders.
        """
        poly_mask_bottom0, poly_mask_top0 = np.copy(self.poly_mask_bottom), np.copy(self.poly_mask_top)
        self.poly_masks = {}
        for order_map in self.data["order_maps"]:
            print(f"Tracing orders for {order_map} ...", flush=True)
            try:
                self.poly_mask_bottom, self.poly_mask_top = self.compute_poly_masks(order_map)
                self.poly_masks[order_map] = [self.poly_mask_bottom, self.poly_mask_top]
            except:
                print(f"Warning! Could not calculate the order map offset from the flat fields with {self.base_flat_field_file}")
            self.tracer.trace(self, order_map)
            with open(f"{self.output_path}{os.sep}trace{os.sep}{order_map.base_input_file_noext}_order_map.pkl", 'wb') as f:
                pickle.dump(order_map.orders_list, f)

            # Reset mask
            self.poly_mask_bottom, self.poly_mask_top = poly_mask_bottom0, poly_mask_top0

    def compute_poly_masks(self, data):
        image1 = pcdata.Echellogram(self.base_flat_field_file, spectrograph="iSHELL").parse_image()
        image2 = data.parse_image()
        offset = self.compute_mask_offset(image1, image2)
        poly_mask_bottom_new, poly_mask_top_new = copy.deepcopy(self.poly_mask_bottom), copy.deepcopy(self.poly_mask_top)
        poly_mask_bottom_new[0][1] -= offset
        poly_mask_bottom_new[1][1] -= offset
        poly_mask_bottom_new[2][1] -= offset
        poly_mask_top_new[0][1] -= offset
        poly_mask_top_new[1][1] -= offset
        poly_mask_top_new[2][1] -= offset
        return poly_mask_bottom_new, poly_mask_top_new

    @staticmethod
    def compute_mask_offset(image1, image2):
        ny, nx = image1.shape
        lags = np.arange(-200, 201)
        n_lags = len(lags)
        n_slices = 100
        ccfs = np.full((n_lags, n_slices), np.nan)
        yarr = np.arange(2048)
        slices = np.linspace(500, ny-500-1, num=n_slices).astype(int)
        for i in range(n_slices):
            ii = slices[i]
            s1 = image1[:, ii] / pcmath.weighted_median(image1[:, ii], percentile=0.95)
            s2 = image2[:, ii] / pcmath.weighted_median(image2[:, ii], percentile=0.95)
            ccfs[:, i] = pcmath.cross_correlate(yarr, s1, yarr, s2, lags, kind='xc')
        ccf = np.nanmedian(ccfs, axis=1)
        lag_best = lags[np.nanargmax(ccf)]
        return lag_best

    def reduce(self):
        """Primary method to reduce a given directory.
        """

        # Start the main clock
        stopwatch = pcutils.StopWatch()

        # Create the output directories
        self.create_output_dirs()

        # init data
        self.init_data()

        # Generate pre calibration images
        pccalib.gen_master_calib_images(self.data, self.do_bias, self.do_dark, self.do_flat, self.flat_percentile)
        
        # Trace orders for appropriate images
        self.trace()

        ny, nx = 2048, 2048

        # Fix in between orders
        for order_map in self.data['order_maps']:
            flat_image = order_map.parse_image()
            order_map_image = self.tracer.gen_image(order_map.orders_list, ny, nx, xrange=self.xrange, poly_mask_top=self.poly_masks[order_map][1], poly_mask_bottom=self.poly_masks[order_map][0])
            bad = np.where(~np.isfinite(order_map_image))
            flat_image[bad] = np.nan
            order_map.save(flat_image)
        
        # Extract all desired images
        self.extract()
        
        # Run Time
        print(f"REDUCTION COMPLETE! TOTAL TIME: {round(stopwatch.time_since() / 3600, 2)} hours")



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
        ny, nx = data_image.shape
        
        # Calibrate image
        pccalib.pre_calibrate(data, data_image, recipe.do_bias, recipe.do_dark, recipe.do_flat)

        # Mask
        if 'badpix_mask' in recipe.data:
            badpix_mask = recipe.data['badpix_mask']
        else:
            badpix_mask = None
        
        # Extract image
        poly_mask_bottom = recipe.poly_mask_bottom
        poly_mask_top = recipe.poly_mask_top
        recipe.poly_mask_bottom = recipe.poly_masks[data.order_maps[0]][0]
        recipe.poly_mask_top = recipe.poly_masks[data.order_maps[0]][1]
        recipe.extractor.extract_image(recipe, data, data_image, badpix_mask=badpix_mask)
        recipe.poly_mask_bottom = poly_mask_bottom
        recipe.poly_mask_top = poly_mask_top
        
        # Print end
        print(f"Extracted {data} in {round(stopwatch.time_since() / 60, 2)} min")