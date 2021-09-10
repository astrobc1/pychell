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
    
    def categorize_raw_data(self, reducer):

        # Stores the data as above objects
        data_dict = {}
        
        # PARVI science files
        all_files = glob.glob(self.input_path + '*data*.fits')
        data_dict['science'] = [pcdata.RawImage(input_file=sci_files[f], parser=self) for f in range(n_sci_files)]
        
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

    def parse_fiber_num(self, data):
        return int(data.header["FIBER"])

    def classify_traces(self, data):
        pass
        # !!HERE!!
        #fibers = data.
        #pass
        
    def parse_spec1d(self, data):
        fits_data = fits.open(data.input_file)
        fits_data.verify('fix')
        data.header = fits_data[0].header
        data.apriori_wave_grid = 10 * fits_data[4].data[0, data.order_num - 1, :]
        data.flux = fits_data[4].data[3, data.order_num - 1, :]
        data.flux_unc = fits_data[4].data[4, data.order_num - 1, :]
        data.mask = np.ones_like(data.flux)
        
    def compute_exposure_midpoint(self, data):
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


#######################################
##### GENERATING RADIAL VELOCITIES ####
#######################################

lsf_linear_coeffs = np.array([0.00167515, 0.08559148])
lsf_widths = np.polyval(lsf_linear_coeffs, np.arange(46))
lsf_widths = [[lw*0.9, lw, lw*1.1] for lw in lsf_widths]