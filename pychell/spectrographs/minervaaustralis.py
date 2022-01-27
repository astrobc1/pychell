import os
import numpy as np
import pychell.rvs

#######################
#### Name and Site ####
#######################

spectrograph = 'MinervaAustralis'

observatory = {
    'name': 'Mt. Kent',
    'lat': -27.7977,
    'lon': 151.8554,
    'alt': 682
}

######################
#### Data Parsing ####
######################

class MinervaAustralisParser(DataParser):
    
    def categorize_raw_data(self, config):

        # Stores the data as above objects
        data_dict = {}
        
        # iSHELL science files are files that contain spc or data
        sci_files1 = glob.glob(self.input_path + '*data*.fits')
        sci_files2 = glob.glob(self.input_path + '*spc*.fits')
        sci_files = sci_files1 + sci_files2
        sci_files = np.sort(np.unique(np.array(sci_files, dtype='<U200'))).tolist()
        n_sci_files = len(sci_files)
        
        data_dict['science'] = [pcdata.RawImage(input_file=sci_files[f], parser=self) for f in range(n_sci_files)]
            
        # Darks assumed to contain dark in filename
        if config['dark_subtraction']:
            dark_files = glob.glob(self.input_path + '*dark*.fits')
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
            flat_files = glob.glob(self.input_path + '*flat*.fits')
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
            thar_files = glob.glob(self.input_path + '*arc*.fits')
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
        
    def parse_time(self, data):
        data.time_obs_start = Time(float(data.header['JD']), scale='utc', format='jd')
        return data.time_obs_start
        
    def get_n_traces(self, data):
        return 1
    
    def get_n_orders(self, data):
        return 29
    
    def parse_itime(self, data):
        data.itime = data.header["ITIME"]
        return data.itime
        
    def parse_spec1d(self, data):
        fits_data = fits.open(data.input_file)[0]
        fits_data.verify('fix')
        data.header = fits_data.header
        oi = data.order_num - 1
        data.default_wave_grid, data.flux, data.flux_unc = fits_data.data[oi, :, 0].astype(np.float64), fits_data.data[oi, :, 1].astype(np.float64), fits_data.data[oi, :, 2].astype(np.float64)
        data.mask = np.ones(data.flux.size)
        
    def compute_barycenter_corrections(self, data, target=None):
        
        # Compute the jd mid point
        jdmid = self.compute_midpoint(data)
        
        # Parse the target
        if target is None:
            target = self.parse_target(data)
        
        # BJD
        data.bjd = JDUTC_to_BJDTDB(JDUTC=jdmid, starname=target.replace('_', ' '), lat=-27.7977, longi=151.8554, alt=682, leap_update=False)[0][0]
        
        # bc vel
        data.bc_vel = get_BC_vel(JDUTC=jdmid, starname=target.replace('_', ' '), lat=-27.7977, longi=151.8554, alt=682, leap_update=False)[0][0]
        
        return data.bjd, data.bc_vel


################################
#### Reduction / Extraction ####
################################

redux_settings = NotImplemented


#######################################
##### GENERATING RADIAL VELOCITIES ####
#######################################

# Default forward model settings
forward_model_settings = {
    
    # The cropped pixels
    'crop_data_pix': [10, 10],
    
    # The units for plotting
    'plot_wave_unit': 'nm',
    
    'observatory': observatory
}



# Forward model blueprints for RVs
forward_model_blueprints = {
    
    # The star
    'star': {
        'name': 'star',
        'class_name': 'Star',
        'input_file': None,
        'vel': [-400000, 0, 400000]
    },
    
    # Tellurics (from TAPAS) NOTE: Still need proper tellurics, so steal Whipple
    'tellurics': {
        'name': 'vis_tellurics',
        'class_name': 'TelluricsTAPAS',
        'vel': [-300, 0, 300],
        'water_depth': [0.01, 1.5, 4.0],
        'airmass_depth': [0.8, 1.2, 4.0],
        'min_range': 0.01,
        'input_files': {
            'water': 'telluric_water_tapas_whipple.npz',
            'methane': 'telluric_methane_tapas_whipple.npz',
            'nitrous_oxide': 'telluric_nitrous_oxide_tapas_whipple.npz',
            'carbon_dioxide': 'telluric_carbon_dioxide_tapas_whipple.npz',
            'oxygen' : 'telluric_oxygen_tapas_whipple.npz',
            'ozone': 'telluric_ozone_tapas_whipple.npz'
        }
    },
    
    # The default blaze is a quadratic + splines.
    'blaze': {
        'name': 'blaze', # The blaze model after a division from a flat field
        'class_name': 'SplineBlaze',
        'n_splines': 5,
        'poly_2': [-5.5E-5, -2E-6, 5.5E-5],
        'poly_1': [-0.001, 1E-5, 0.001],
        'poly_0': [0.96, 1.0, 1.08],
        'spline': [0.2, 0.8, 1.1],
        
        # Blaze is centered on the blaze wavelength. Crude estimates unless using a full blaze model
        'blaze_wavelengths': [4858.091694040058, 4896.964858182707, 4936.465079384465, 4976.607650024426, 5017.40836614558, 5058.88354743527, 5101.050061797753, 5143.9253397166585, 5187.527408353689, 5231.87491060088, 5276.98712989741, 5322.884028578407, 5369.586262921349, 5417.11522691744, 5465.493074938935, 5514.742760771861, 5564.888075329751, 5615.953682999512, 5667.96515950171, 5720.949036590132, 5774.932851929652, 5829.94518764045, 5886.015725989253, 5943.1753026380065, 6001.455961651197, 6060.891016560821, 6121.515108109428, 6183.364282120176, 6246.47605505618]
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class_name': 'HermiteLSF',
        'hermdeg': 0,
        'n_delay': 1,
        'width': [0.0234, 0.0234, 0.0234], # LSF width, in angstroms
        'ak': [-0.03, 0.001, 0.2] # Hermite polynomial coefficients
    },
    
    # Quadratic (Lagrange points) + splines
    'wavelength_solution': {
        'name': 'wavesol_ThAr_I2',
        'class_name': 'HybridWavelengthSolution',
        'n_splines': 0, # Zero until I2 cell is implemented
        'spline': [-0.03, 0.0005, 0.03]
    }
}