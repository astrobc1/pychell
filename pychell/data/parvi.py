# Base Python
import os

from pychell.data.parser import DataParser
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

spectrograph = "PARVI"
observatory = {
    "name" : "Palomar",
    "lat": 33.3537819182,
    "long": -116.858929898,
    "alt": 1713.0
}

######################
#### DATA PARSING ####
######################

class PARVIParser(DataParser):
    
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
    
    def parse_itime(self, data):
        data.itime = data.header["EXPTIME"]
        return data.itime
        
    def parse_exposure_start_time(self, data):
        data.time_obs_start = Time(float(data.header["START"]) / 1E9, format="unix")
        return data.time_obs_start
        
    def get_n_traces(self, data):
        return 2
    
    def get_n_orders(self, data):
        mode = data.header["XDTILT"].lower()
        if mode == "kgas":
            return 29
        else:
            return None
        
    def parse_spec1d(self, data):
        fits_data = fits.open(data.input_file)
        fits_data.verify('fix')
        data.header = fits_data[0].header
        
        # For GJ 229 formatted data (old?)
        #data.apriori_wave_grid = 10 * fits_data[1].data[0, data.order_num - 1, :]
        #data.flux = fits_data[1].data[7, data.order_num - 1, :]
        #data.flux_unc = fits_data[1].data[8, data.order_num - 1, :]
        #data.mask = np.ones_like(data.flux)
        
        # For Tau Boo formatted data (June 2021) (is this the new standard?)
        data.apriori_wave_grid = 10 * fits_data[4].data[0, data.order_num - 1, :]
        data.flux = fits_data[4].data[3, data.order_num - 1, :]
        data.flux_unc = fits_data[4].data[4, data.order_num - 1, :]
        data.mask = np.ones_like(data.flux)
        
    def compute_midpoint(self, data):
        jds, fluxes = [], []
        # Eventually we will fill fluxes with an arbitrary read value.
        # Then, the mean_jd will be computed with pcmath.weighted_mean(jds[1:], np.diff(fluxes))
        for key in data.header:
            if key.startswith("TIMEI"):
                jds.append(Time(float(data.header[key]) / 1E9, format="unix").jd)
        jds = np.array(jds)
        mean_jd = np.nanmean(jds)
        return mean_jd



################################
#### REDUCTION / EXTRACTION ####
################################

redux_settings = NotImplemented


#######################################
##### GENERATING RADIAL VELOCITIES ####
#######################################

lsf_width = [0.05, 0.08, 0.12]