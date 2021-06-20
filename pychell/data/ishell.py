# Base Python
import os

from pychell.data.parser import DataParser
import glob
from astropy.io import fits
import pychell.data.spectraldata as pcdata

# Maths
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as units

# Pychell deps
import pychell.maths as pcmath


#######################
#### NAME AND SITE ####
#######################

spectrograph = 'iSHELL'
observatory = {
    'name': 'IRTF',
    'lat': 19.826218316666665,
    'lon': -155.4719987888889,
    'alt': 4168.066848
}

######################
#### DATA PARSING ####
######################

class iSHELLParser(DataParser):
    
    def categorize_raw_data(self, config):

        # Stores the data as above objects
        data_dict = {}
        
        # iSHELL science files are files that contain spc or data
        sci_files1 = glob.glob(self.data_input_path + '*data*.fits')
        sci_files2 = glob.glob(self.data_input_path + '*spc*.fits')
        sci_files = sci_files1 + sci_files2
        sci_files = np.sort(np.unique(np.array(sci_files, dtype='<U200'))).tolist()
        n_sci_files = len(sci_files)
        
        data_dict['science'] = [pcdata.RawImage(input_file=sci_files[f], parser=self) for f in range(n_sci_files)]
            
        # Darks assumed to contain dark in filename
        if config['dark_subtraction']:
            dark_files = glob.glob(self.data_input_path + '*dark*.fits')
            n_dark_files = len(dark_files)
            data_dict['darks'] = [pcdata.RawImage(input_file=dark_files[f], parser=self) for f in range(n_dark_files)]
            dark_groups = self.group_darks(data_dict['darks'])
            data_dict['master_darks'] = []
            for dark_group in dark_groups:
                master_dark_fname = self.gen_master_dark_filename(dark_group)
                data_dict['master_darks'].append(pcdata.MasterCalibImage(dark_group, input_file=master_dark_fname, parser=self))
            
            for sci in data_dict['science']:
                self.pair_master_dark(sci, data_dict['master_darks'])
                
            for flat in data_dict['flats']:
                self.pair_master_dark(flat, data_dict['master_darks'])
        
        # iSHELL flats must contain flat in the filename
        if config['flat_division']:
            flat_files = glob.glob(self.data_input_path + '*flat*.fits')
            n_flat_files = len(flat_files)
            data_dict['flats'] = [pcdata.RawImage(input_file=flat_files[f], parser=self) for f in range(n_flat_files)]
            flat_groups = self.group_flats(data_dict['flats'])
            data_dict['master_flats'] = []
            for flat_group in flat_groups:
                master_flat_fname = self.gen_master_flat_filename(flat_group)
                data_dict['master_flats'].append(pcdata.MasterCalibImage(flat_group, input_file=master_flat_fname, parser=self))
            
            for sci in data_dict['science']:
                self.pair_master_flat(sci, data_dict['master_flats'])
            
        # iSHELL ThAr images must contain arc (not implemented yet)
        if config['wavelength_calibration']:
            thar_files = glob.glob(self.data_input_path + '*arc*.fits')
            data_dict['wavecals'] = [pcdata.RawImage(input_file=thar_files[f], parser=self) for f in range(len(thar_files))]
            data_dict['master_wavecals'] = self.group_wavecals(data_dict['wavecals'])
            for sci in data_dict['science']:
                self.pair_master_wavecal(sci, data_dict['master_wavecals'])
                
        # Order map
        data_dict['order_maps'] = []
        for master_flat in data_dict['master_flats']:
            order_map_fname = self.gen_order_map_filename(source=master_flat)
            data_dict['order_maps'].append(pcdata.ImageMap(input_file=order_map_fname, source=master_flat,  parser=self, order_map_fun='trace_orders_from_flat_field'))
        for sci_data in data_dict['science']:
            self.pair_order_map(sci_data, data_dict['order_maps'])
        
        self.print_summary(data_dict)

        return data_dict
    
    def pair_order_map(self, data, order_maps):
        for order_map in order_maps:
            if order_map.source == data.master_flat:
                data.order_map = order_map
                return

    def parse_image_num(self, data):
        string_list = data.base_input_file.split('.')
        data.image_num = string_list[4]
        return data.image_num
        
    def parse_itime(self, data):
        data.itime = data.header["ITIME"]
        return data.itime    
    
    def parse_target(self, data):
        data.target = data.header["OBJECT"]
        return data.target
        
    def parse_utdate(self, data):
        utdate = "".join(data.header["DATE_OBS"].split('-'))
        data.utdate = utdate
        return data.utdate
        
    def parse_sky_coord(self, data):
        data.skycoord = SkyCoord(ra=data.header['TCS_RA'], dec=data.header['TCS_DEC'], unit=(units.hourangle, units.deg))
        return data.skycoord
        
    def parse_exposure_start_time(self, data):
        data.time_obs_start = Time(float(data.header['TCS_UTC']) + 2400000.5, scale='utc', format='jd')
        return data.time_obs_start
        
    def get_n_traces(self, data):
        return 1
    
    def get_n_orders(self, data):
        mode = data.header["XDTILT"].lower()
        if mode == "kgas":
            return 29
        else:
            return None
        
    def parse_spec1d(self, data):
        
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

    #########################
    #### BASIC WAVE INFO ####
    #########################
    
    def estimate_wavelength_solution(self, data):
        oi = data.order_num - 1
        waves = np.array([quad_set_point_1[oi], quad_set_point_2[oi], quad_set_point_3[oi]])
        pcoeffs = pcmath.poly_coeffs(quad_pixel_set_points, waves)
        wls = np.polyval(pcoeffs, np.arange(data.flux.size))
        return wls
        

################################
#### REDUCTION / EXTRACTION ####
################################

redux_settings = {
    
    # Detector properties
    'detector_props' : [{'gain': 1.8, 'dark_current': 0.05, 'read_noise': 8.0}],
    
    # Calibration
    'dark_subtraction': False,
    'flat_division': True,
    'bias_subtraction': False,
    'wavelength_calibration': False,
    'flatfield_percentile': 0.75,
    
    # Pixels to mask on the top, bottom, left, and right edges
    'mask_image_left': 200,
    'mask_image_right': 200,
    'mask_image_top': 20,
    'mask_image_bottom': 20,
    
    # How many pixels to flag on the top/bottom of the trace profile
    # The trace profile is by default cropped.
    'mask_trace_edges':  3,
    'min_profile_flux': 0.05,
    
    # The degree of the polynomial to fit the individual order locations
    'trace_pos_polyorder' : 2,
    
    # Whether or not to perform a sky subtraction
    # The number of rows used to estimate the sky background (lowest n_sky_rows in the trace profile are used).
    'sky_subtraction': True,
    'n_sky_rows': 8,
    
    # The trace profile is constructed using oversampled data.
    # This is the oversample factor.
    'oversample': 4,
    
    # The optimal extraction algorithm
    'optx_alg': 'pmassey_wrapper',
    'order_map': {'source': 'empirical_from_flat_fields', 'method': None}
}


#######################################
##### GENERATING RADIAL VELOCITIES ####
#######################################

# Forward model blueprints for RVs
# No default blueprints are defined.
spectral_model_blueprints = {
    
    # The star
    'star': {
        'class': 'AugmentedStar',
        'input_file': None,
        'vel': [-1000 * 300, 10, 1000 * 300]
    },
    
    # The methane gas cell
    'gas_cell': {
        'name': 'methane_gas_cell', # NOTE: full parameter names are name + base_name.
        'class': 'DynamicGasCell',
        'input_file': 'methane_gas_cell_ishell_kgas.npz',
        'shift': [0, 0, 0],
        'depth': [0.97, 0.97, 0.97]
    },
    
    # Tellurics (from TAPAS)
    'tellurics': {
        'name': 'kband_tellurics',
        'class': 'TelluricsTAPAS',
        'vel': [-500, -100, 500],
        'water_depth': [0.01, 1.2, 5.0],
        'airmass_depth': [0.8, 1.2, 4.0],
        "feature_depth": 0.02,
        'input_files': {
            'water': 'telluric_water_tapas_maunakea.npz',
            'methane': 'telluric_methane_tapas_maunakea.npz',
            'nitrous_oxide': 'telluric_nitrous_oxide_tapas_maunakea.npz',
            'carbon_dioxide': 'telluric_carbon_dioxide_tapas_maunakea.npz',
            'oxygen' : 'telluric_oxygen_tapas_maunakea.npz',
            'ozone': 'telluric_ozone_tapas_maunakea.npz'
        }
    },
    
    # The default blaze is a quadratic + splines.
    'continuum': {
        'name': 'residual_blaze', # The blaze model after a division from a flat field
        'class': 'SplineContinuum',
        'n_splines': 10,
        'poly_order': 2,
        'poly_6': [-5.5E-9, -2E-8, 5.5E-9],
        'poly_5': [-5.5E-8, -2E-7, 5.5E-8],
        'poly_4': [-5.5E-7, -8E-6, 5.5E-7],
        'poly_3': [-5.5E-6, -4E-6, 5.5E-6],
        'poly_2': [-5.5E-5, -2E-6, 5.5E-5],
        'poly_1': [-0.001, 1E-5, 0.001],
        'poly_0': [0.96, 1.0, 1.1],
        'spline_lagrange': [0.3, 0.95, 1.2]
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class': 'HermiteLSF',
        'hermdeg': 6,
        "nx": 256,
        'width': [0.08, 0.11, 0.15], # LSF width, in angstroms
        #'width': [0.11037, 0.11037, 0.11037],
        'ak': [-0.1, 0.001, 0.1] # Hermite polynomial coefficients
    },
    
    # Determined by splines
    'wavelength_solution': {
        'name': 'csplines_wavesol',
        'class': 'SplineWavelengthSolution',
        'poly_order': 6,
        'n_splines': 6,
        'poly_wave_lagrange': [-0.35, 0.01, 0.35],
        'spline_lagrange': [-0.35, 0.01, 0.35]
    }
}


###############
#### MISC. ####
###############


# Information to generate a crude ishell wavelength solution for the above method estimate_wavelength_solution
quad_pixel_set_points = [199, 1023.5, 1847]

# Left most set point for the quadratic wavelength solution
quad_set_point_1 = [24545.57561435, 24431.48444449, 24318.40830764, 24206.35776048, 24095.33986576, 23985.37381209, 23876.43046386, 23768.48974584, 23661.54443537, 23555.56359209, 23450.55136357, 23346.4923953, 23243.38904298, 23141.19183839, 23039.90272625, 22939.50127095, 22840.00907242, 22741.40344225, 22643.6481698, 22546.74892171, 22450.70934177, 22355.49187891, 22261.08953053, 22167.42305394, 22074.72848136, 21982.75611957, 21891.49178289, 21801.07332421, 21711.43496504]

# Middle set point for the quadratic wavelength solution
quad_set_point_2 = [24628.37672608, 24513.79686837, 24400.32734124, 24287.85495107, 24176.4424356, 24066.07880622, 23956.7243081, 23848.39610577, 23741.05658955, 23634.68688897, 23529.29771645, 23424.86836784, 23321.379387, 23218.80573474, 23117.1876433, 23016.4487031, 22916.61245655, 22817.65768889, 22719.56466802, 22622.34315996, 22525.96723597, 22430.41612825, 22335.71472399, 22241.83394135, 22148.73680381, 22056.42903627, 21964.91093944, 21874.20764171, 21784.20091295]

# Right most set point for the quadratic wavelength solution
quad_set_point_3 = [24705.72472863, 24590.91231465, 24476.99298677, 24364.12010878, 24252.31443701, 24141.55527091, 24031.82506843, 23923.12291214, 23815.40789995, 23708.70106907, 23602.95596074, 23498.18607941, 23394.35163611, 23291.44815827, 23189.49231662, 23088.42080084, 22988.26540094, 22888.97654584, 22790.57559244, 22693.02942496, 22596.33915038, 22500.49456757, 22405.49547495, 22311.25574559, 22217.91297633, 22125.33774808, 22033.50356525, 21942.41058186, 21852.24253555]