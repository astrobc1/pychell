# Base Python
import os
import importlib
import copy

from pychell.data.parser import DataParser
import glob
from astropy.io import fits
import pychell.data.spectraldata as pcspecdata

# Maths
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as units

# Pychell deps
import pychell.maths as pcmath


#############################
####### Name and Site #######
#############################

spectrograph = 'NIRSPEC'
observatory = {
    "name": "Keck"
}

class NIRSPECParser(DataParser):
    
    def categorize_raw_data(self, reducer):

        # Stores the data as above objects
        data = {}
        
        # iSHELL science files are files that contain spc or data
        sci_files = glob.glob(reducer.data_input_path + "*data*.fits")
        data['science'] = [pcspecdata.RawEchellogram(input_file=sci_file, parser=self) for sci_file in sci_files]

        if reducer.pre_calib is not None:
            
            # Darks assumed to contain dark in filename
            if reducer.pre_calib.do_dark:
                dark_files = glob.glob(reducer.data_input_path + '*dark*.fits')
                data['darks'] = [pcspecdata.RawEchellogram(input_file=dark_files[f], parser=self) for f in range(len(dark_files))]
                dark_groups = self.group_darks(data['darks'])
                data['master_darks'] = [pcspecdata.MasterCal(dark_group, reducer.output_path + "calib" + os.sep) for dark_groups in dark_group]
                
                for sci in data['science']:
                    self.pair_master_dark(sci, data['master_darks'])
                    
                for flat in data['flats']:
                    self.pair_master_dark(flat, data['master_darks'])
        
            # iSHELL flats must contain flat in the filename
            if reducer.pre_calib.do_flat:
                flat_files = glob.glob(reducer.data_input_path + '*flat*.fits')
                n_flat_files = len(flat_files)
                data['flats'] = [pcspecdata.RawEchellogram(input_file=flat_files[f], parser=self) for f in range(n_flat_files)]
                flat_groups = self.group_flats(data['flats'])
                data['master_flats'] = [pcspecdata.MasterCal(flat_group, reducer.output_path + "calib" + os.sep) for flat_group in flat_groups]
                
                for sci in data['science']:
                    self.pair_master_flat(sci, data['master_flats'])
            
        # iSHELL ThAr images must contain arc (not implemented yet!)
        # if reducer.wave_cal is not None:
        #     thar_files = glob.glob(reducer.data_input_path + '*arc*.fits')
        #     data['wavecals'] = [pcspecdata.RawImage(input_file=thar_files[f], parser=self) for f in range(len(thar_files))]
        #     wavecal_groups = self.group_wavecals(data['wavecals'])
                
        # Order maps for iSHELL are the flat fields closest in time and space (RA+Dec) to the science target
        data['order_maps'] = data['master_flats']
        for sci_data in data['science']:
            self.pair_order_maps(sci_data, data['order_maps'])

        # Which to extract
        data['extract'] = data['science']
        

        # Print reduction summary
        self.print_reduction_summary(data)

        # Return the data dict
        return data
    
    def pair_order_maps(self, data, order_maps):
        for order_map in order_maps:
            if order_map == data.master_flat:
                data.order_maps = [order_map]

    def parse_image_num(self, data):
        string_list = data.base_input_file.split('.')
        data.image_num = string_list[1]
        return data.image_num
        
    def parse_itime(self, data):
        data.itime = data.header["ITIME"]
        return data.itime
    
    def parse_object(self, data):
        data.object = data.header["OBJECT"].replace(" ", "")
        return data.object
        
    def parse_utdate(self, data):
        utdate = "".join(data.header["DATE-OBS"].split('-'))
        data.utdate = utdate
        return data.utdate
        
    def parse_sky_coord(self, data):
        data.skycoord = SkyCoord(ra=data.header['RA'], dec=data.header['DEC'], unit=(units.hourangle, units.deg))
        return data.skycoord
        
    def parse_exposure_start_time(self, data):
        data.time_obs_start = Time(float(data.header['MJD-OBS']) + 2400000.5, scale='utc', format='jd')
        return data.time_obs_start

    def gen_master_calib_filename(self, master_cal):
        fname0 = master_cal.group[0].base_input_file.lower()
        if "dark" in fname0:
            return f"master_dark_{master_cal.group[0].utdate}{group[0].itime}s.fits"
        elif "flat" in fname0:
            img_nums = np.array([self.parse_image_num(d) for d in master_cal.group], dtype=int)
            img_start, img_end = img_nums.min(), img_nums.max()
            return f"master_flat_{master_cal.group[0].utdate}imgs{img_start}-{img_end}.fits"
        else:
            return f"master_calib_{master_cal.group[0].utdate}.fits"

    def gen_master_calib_header(self, master_cal):
        master_cal.skycoord = master_cal.group[0].skycoord
        master_cal.time_obs_start = master_cal.group[0].time_obs_start
        master_cal.object = master_cal.group[0].object
        master_cal.itime = master_cal.group[0].itime
        return copy.deepcopy(master_cal.group[0].header)
        
    def parse_spec1d(self, data):
        
        # Load the flux, flux unc, and bad pix arrays
        fits_data = fits.open(data.input_file, output_verify='ignore')[0]
        data.header = fits_data.header
        oi = data.order_num - 1
        data.flux, data.flux_unc, data.mask = fits_data.data[oi, 0, :, 0].astype(np.float64), fits_data.data[oi, 0, :, 1].astype(np.float64), fits_data.data[oi, 0, :, 2].astype(np.float64)

    def parse_image(self, data):
        image = fits.open(data.input_file, do_not_scale_image_data=True)[0].data.astype(float)
        self.correct_readmath(data, image)
        return image


    #########################
    #### BASIC WAVE INFO ####
    #########################
    
    def estimate_wavelength_solution(self, data):
        oi = data.order_num - 1
        waves = np.array([quad_set_point_1[oi], quad_set_point_2[oi], quad_set_point_3[oi]])
        pcoeffs = pcmath.poly_coeffs(quad_pixel_set_points, waves)
        wls = np.polyval(pcoeffs, np.arange(data.flux.size))
        return wls

read_noise = 0

###########################
#### RADIAL VELOCITIES ####
###########################

lsf_width = [0.1, 0.25, 0.4]
rv_zero_point = 0

# Information to generate a crude ishell wavelength solution for the above method estimate_wavelength_solution
quad_pixel_set_points = [1, 512, 1023]

# Left most set point for the quadratic wavelength solution
quad_set_point_1 = np.array([19900.00 - 36, 20400.00 + 2, 20600.00,21500.00,22200.00,22800.00,23600.00])

# Middle set point for the quadratic wavelength solution
quad_set_point_2 = np.array([20050.00 - 40.5, 20550.00 + 2, 20950.00, 21600.00, 22350.00, 23000.00, 23750.00])

# Right most set point for the quadratic wavelength solution
quad_set_point_3 = np.array([20200.00 - 40, 20700.00 + 2, 21300.00, 21700.00, 22500.00, 23200.00, 23900.00])