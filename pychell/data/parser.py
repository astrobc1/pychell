# Base Python
import warnings
import glob
import os
import importlib
import copy
import sys
import pickle

# Import the barycorrpy module
try:
    from barycorrpy import get_BC_vel
    from barycorrpy.utc_tdb import JDUTC_to_BJDTDB
except:
    warnings.warn("Could not import barycorrpy")

# Astropy
from astropy.time import Time
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as units

# Maths
import sklearn.cluster
import numpy as np

# Pychell
import pychell.data as pcdata
import pychell.maths as pcmath


class DataParser:
    """Base class for parsing/generating information from spectrograph specific data files.
    """
    
    #####################
    #### CONSTRUCTOR ####
    #####################
    
    def __init__(self, data_input_path, output_path=None):
        """Construct a parser object.

        Args:
            data_input_path (str): The full path to the data to be parsed.
            output_path (str, optional): The output path for writing any calibration files, only used for the reduce module. Defaults to None.
        """
        self.data_input_path = data_input_path
        self.output_path = output_path
    
    #############################
    #### CATEGORIZE RAW DATA ####
    #############################
    
    def categorize_raw_data(self):
        # Allowed entries in data_dict:
        # flats, darks, bias
        # master_flats, master_darks, master_bias
        # science
        # wavecals
        # skycals
        raise NotImplementedError(f"Must implement a categorize_raw_data method for class {self.__class__.__name__}")
    
    def categorize_traces(self, data):
        raise NotImplementedError(f"Must implement a categorize_traces method for class {self.__class__.__name__}")
    
    ################################
    #### PARSE OBSERVATION INFO ####
    ################################
    
    def parse_image_header(self, data):
        
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
        self.parse_sky_coord(data)
        self.parse_exposure_start_time(data)
        self.parse_target(data)
        self.parse_itime(data)
        
        return data.header
    
    def parse_itime(self, data):
        raise NotImplementedError(f"Must implement a parse_itime method for class {self.__class__.__name__}")
    
    def parse_utdate(self, data):
        raise NotImplementedError(f"Must implement a parse_local_date method for class {self.__class__.__name__}")
    
    def parse_exposure_start_time(self, data):
        raise NotImplementedError(f"Must implement a parse_time method for class {self.__class__.__name__}")
    
    def parse_image_num(self, data):
        raise NotImplementedError(f"Must implement a parse_image_num method for class {self.__class__.__name__}")
    
    def parse_sky_coord(self, data):
        raise NotImplementedError(f"Must implement a parse_sky_coord method for class {self.__class__.__name__}")
    
    def parse_image(self, data):
        image = fits.open(data.input_file, do_not_scale_image_data=True)[0].data.astype(float)
        return image
    
    #####################
    #### CALIBRATION ####
    #####################
    
    def gen_master_dark_filename(self, group):
        fname = f"{self.output_path}{os.sep} calib {os.sep}master_dark_{group[0].utdate}{group[0].itime}s.fits"
        return fname
    
    def gen_master_flat_filename(self, group):
        img_nums = np.array([int(f.image_num) for f in group])
        img_start, img_end = str(np.min(img_nums)), str(np.max(img_nums))
        fname = f"{self.output_path}calib{os.sep}master_flat_{group[0].utdate}_imgs{img_start}-{img_end}.fits"
        return fname
    
    def gen_order_map_filename(self, source):
        fname = f"{self.output_path}calib{os.sep}{source.base_input_file_noext}_order_map.fits"
        return fname
    
    def gen_master_calib_header(self, data):
        data.header = copy.deepcopy(data.individuals[0].header)
        data.skycoord = copy.deepcopy(data.individuals[0].skycoord)
        data.time_obs_start = copy.deepcopy(data.individuals[0].time_obs_start)
    
    def pair_master_dark(self, data, master_darks):
        n_masker_darks = len(master_darks)
        itimes = np.array([master_darks[i].itime for i in range(n_masker_darks)], dtype=float)
        good = np.where(data.itime == itimes)[0]
        if good.size != 1:
            raise ValueError(str(good.size) + " master dark(s) found for\n" + str(self))
        else:
            data.master_dark = master_darks[good[0]]
    
    def pair_master_flat(self, data, master_flats):
        ang_seps = np.array([np.abs(data.skycoord.separation(master_flat.skycoord)).value for master_flat in master_flats], dtype=float)
        time_seps = np.array([np.abs(data.time_obs_start.value - master_flat.time_obs_start.value) for master_flat in master_flats], dtype=float)
        ds = np.sqrt(ang_seps**2 + time_seps**2)
        minds_loc = np.argmin(ds)
        data.master_flat = master_flats[minds_loc]
    
    def group_darks(self, darks):
        groups = []
        itimes = np.array([dark.itime for dark in darks])
        itimes_unq = np.unique(itimes)
        for t in itimes_unq:
            good = np.where(itimes == t)[0]
            indiv_darks = [darks[i] for i in good]
            groups.append(indiv_darks)
        return groups
    
    def group_flats(self, flats):
        
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
                dpsi = np.abs(flats[i].skycoord.separation(flats[j].skycoord).value)
                dt = np.abs(flats[i].time_obs_start.jd - flats[j].time_obs_start.jd)
                dpsi /= np.pi
                dt /= 10  # Places more emphasis on delta psi
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
    
    def pair_master_bias(self, data, master_bias):
        data.master_bias = master_bias
        
    def pair_order_map(self, data):
        pass
        
    #########################
    #### BASIC WAVE INFO ####
    #########################
    
    def estimate_wavelength_solution(self, data):
        if hasattr(data, "apriori_wave_grid"):
            return data.apriori_wave_grid
        
    #########################
    #### MISC. REDUCTION ####
    #########################
    
    def correct_readmath(self, data, data_image):
        # Corrects NDRs - Number of dynamic reads, or Non-destructive reads, take your pick.
        # This reduces the read noise by sqrt(NDR)
        if hasattr(data, "header"):
            if 'NDR' in data.header:
                data_image /= float(data.header['NDR'])
    
    def save_reduced_orders(self, data, reduced_data):
        fname = f"{self.output_path}spectra{os.sep}{data.base_input_file_noext}_{data.target}_reduced.fits"
        hdu = fits.PrimaryHDU(reduced_data, header=data.header)
        hdu.writeto(fname, overwrite=True)
    
    ###################################
    #### BARYCENTENTER CORRECTIONS ####
    ###################################
    
    def compute_barycenter_corrections(self, data, observatory, target_dict):
        
        # Star name
        star_name = target_dict["name"].replace('_', ' ')
        
        # Compute the jd mid point
        jdmid = self.compute_exposure_midpoint(data)
        
        # BJD
        bjd = JDUTC_to_BJDTDB(JDUTC=jdmid, starname=star_name, obsname=observatory['name'], leap_update=False)[0][0]
        
        # bc vel
        bc_vel = get_BC_vel(JDUTC=jdmid, starname=star_name, obsname=observatory['name'], leap_update=False)[0][0]
        
        # Add to data
        data.bjd = bjd
        data.bc_vel = bc_vel
        
        return bjd, bc_vel
    
    def compute_exposure_midpoint(self, data):
        return self.parse_exposure_start_time(data).jd + self.parse_itime(data) / (2 * 86400)
        
    ###############
    #### MISC. ####
    ###############
    
    def get_n_orders(self, data):
        raise NotImplementedError(f"Must implement a get_n_orders method for class {self.__class__.__name__}")
            
    def print_summary(self, data_dict):
    
        n_sci_tot = len(data_dict['science'])
        targets_all = np.array([data_dict['science'][i].target for i in range(n_sci_tot)], dtype='<U50')
        targets_unique = np.unique(targets_all)
        for i in range(len(targets_unique)):
            
            target = targets_unique[i]
            
            locs_this_target = np.where(targets_all == target)[0]
            
            sci_this_target = [data_dict['science'][j] for j in locs_this_target]
            
            print('Target: ' + target)
            print('    N Exposures: ' + str(locs_this_target.size))
            if hasattr(sci_this_target[0], 'master_bias'):
                print('    Master Bias File(s): ')
                print('    ' + data_dict['science'].master_bias.base_input_file)
                
            if hasattr(sci_this_target[0], 'master_dark'):
                darks_this_target_all = np.array([sci.master_dark for sci in sci_this_target], dtype=pcdata.RawImage)
                darks_unique = np.unique(darks_this_target_all)
                print('  Master Dark File(s): ')
                for d in darks_unique:
                    print('    ' + d.base_input_file)
                
            if hasattr(sci_this_target[0], 'master_flat'):
                flats_this_target_all = np.array([sci.master_flat for sci in sci_this_target], dtype=pcdata.RawImage)
                flats_unique = np.unique(flats_this_target_all)
                print('  Master Flat File(s): ')
                for f in flats_unique:
                    print('    ' + f.base_input_file)
                    
            print('')
