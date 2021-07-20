# Base Python
import os

import pychell.data.parser as pcdataparser
import glob
from astropy.io import fits
import pychell.data as pcdata

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

spectrograph = 'CHIRON'
observatory = {
    "name": 'CTIO',
    "lat": 30.169286111111113,
    "long": 70.806789,
    "alt": 2207
}

######################
#### DATA PARSING ####
######################

class CHIRONParser(pcdataparser.DataParser):
    
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
        utdate = "".join(data.header["DATE-OBS"].split('-')[0:3])
        data.utdate = utdate
        return data.utdate
        
    def parse_sky_coord(self, data):
        data.skycoord = SkyCoord(ra=data.header['RA'], dec=data.header['DEC'], unit=(units.hourangle, units.deg))
        return data.skycoord
        
    def parse_exposure_start_time(self, data):
        data.time_obs_start = Time(data.header['DATE-OBS'])
        return data.time_obs_start
        
    def get_n_traces(self, data):
        return 1
    
    def get_n_orders(self, data):
        return 62
    
    def parse_itime(self, data):
        data.itime = data.header["EXPTIME"]
        return data.itime
        
    def parse_spec1d(self, data):
        fits_data = fits.open(data.input_file)[0]
        fits_data.verify('fix')
        data.header = fits_data.header
        oi = data.order_num - 1
        data.apriori_wave_grid, data.flux = fits_data.data[oi, :, 0].astype(np.float64), fits_data.data[oi, :, 1].astype(np.float64)
        data.flux_unc = np.zeros_like(data.flux) + 1E-3
        data.mask = np.ones_like(data.flux)
        
    def estimate_wavelength_solution(self, data):
        oi = data.order_num - 1
        shift = gas_cell_shifts[oi]
        wls = data.apriori_wave_grid - shift
        return wls


#######################################
##### GENERATING RADIAL VELOCITIES ####
#######################################

lsf_width = [0.009, 0.014, 0.018]

###############
#### MISC. ####
###############

# Shifts between the thar lamp and gas cell for each order
gas_cell_shifts = [-1.28151621, -1.28975381, -1.29827329, -1.30707465, -1.31615788, -1.32552298, -1.33516996, -1.34509881, -1.35530954, -1.36580215, -1.37657662, -1.38763298, -1.3989712, -1.4105913, -1.42249328, -1.43467713, -1.44714286, -1.45989046, -1.47291993, -1.48623128, -1.49982451, -1.5136996 , -1.52785658, -1.54229543, -1.55701615, -1.57201875, -1.58730322, -1.60286957, -1.61871779, -1.63484788, -1.65125985, -1.6679537 , -1.68492942, -1.70218701, -1.71972648, -1.73754783, -1.75565104, -1.77403614, -1.79270311, -1.81165195, -1.83088267, -1.85039526, -1.87018972, -1.89026606, -1.91062428, -1.93126437, -1.95218634, -1.97339018, -1.99487589, -2.01664348, -2.03869294, -2.06102428, -2.08363749, -2.10653258, -2.12970954, -2.15316838, -2.17690909, -2.20093168, -2.22523614, -2.24982247, -2.27469068, -2.29984077, -2.32527273]