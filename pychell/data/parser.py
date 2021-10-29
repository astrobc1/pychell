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
import pychell.data.spectraldata as pcspecdata
import pychell.maths as pcmath


class DataParser:
    """Base class for parsing/generating information from spectrograph specific data files.
    """
    
    #############################
    #### CATEGORIZE RAW DATA ####
    #############################
    
    def categorize_raw_data(self, reducer):
        raise NotImplementedError(f"Must implement a categorize_raw_data method for class {self.__class__.__name__}")
    

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
        self.parse_object(data)
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

    def parse_object(self, data):
        raise NotImplementedError(f"Must implement parse_object method for class {self.__class__.__name__}")

    def parse_fiber_nums(self, data):
        return None

    #####################
    #### CALIBRATION ####
    #####################

    def gen_master_calib_filename(self, master_cal):
        raise NotImplementedError(f"Must implement method gen_master_calib_filename for class {self.__class__.__name__}")

    def gen_master_calib_header(self, master_cal):
        return copy.deepcopy(master_cal.group[0].header)

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
    
    def estimate_wls(self, data):
        if hasattr(data, "wave"):
            return data.apriori_wave_grid
        else:
            raise NotImplementedError(f"Must implement a method estimate_wls for {self.__class__.__name__} or provide the attribute 'wave' with the data")


    #########################
    #### MISC. REDUCTION ####
    #########################
    
    def correct_readmath(self, data, data_image):
        # Corrects NDRs - Number of dynamic reads, or Non-destructive reads, take your pick.
        # This reduces the read noise by sqrt(NDR)
        if hasattr(data, "header"):
            if 'NDR' in data.header:
                data_image /= float(data.header['NDR'])
            if 'BZERO' in data.header:
                data_image -= float(data.header['BZERO'])
            if 'BSCALE' in data.header:
                data_image /= float(data.header['BSCALE'])
    
    def compile_reduced_outputs(self, reducer):
        pass

    ###################################
    #### BARYCENTENTER CORRECTIONS ####
    ###################################
    
    def compute_barycenter_corrections(self, data, observatory, star_name):
        
        # Star name
        star_name = star_name.replace('_', ' ')
        
        # Compute the JD UTC mid point (possibly weighted)
        jdmid = self.compute_exposure_midpoint(data)
        
        # BJD
        bjd = JDUTC_to_BJDTDB(JDUTC=jdmid, starname=star_name, obsname=observatory['name'], leap_update=True)[0][0]
        
        # bc vel
        bc_vel = get_BC_vel(JDUTC=jdmid, starname=star_name, obsname=observatory['name'], leap_update=True)[0][0]
        
        # Add to data
        data.bjd = bjd
        data.bc_vel = bc_vel
        
        return bjd, bc_vel
    
    def compute_exposure_midpoint(self, data):
        return self.parse_exposure_start_time(data).jd + self.parse_itime(data) / (2 * 86400)


    ###############
    #### MISC. ####
    ###############

    def compile_reduced_outputs(self, reducer):
        pass
            
    def print_reduction_summary(self, data):
    
        n_sci_tot = len(data['science'])
        targets_all = np.array([data['science'][i].object for i in range(n_sci_tot)], dtype='<U50')
        targets_unique = np.unique(targets_all)
        for i in range(len(targets_unique)):
            
            target = targets_unique[i]
            
            locs_this_target = np.where(targets_all == target)[0]
            
            sci_this_target = [data['science'][j] for j in locs_this_target]
            
            print('Target: ' + target)
            print('    N Exposures: ' + str(locs_this_target.size))
            if hasattr(sci_this_target[0], 'master_bias'):
                print('  Master Bias: ')
                print('    ' + str(sci_this_target[0].master_bias))
                
            if hasattr(sci_this_target[0], 'master_dark'):
                print('  Master Dark: ')
                print('    ' + str(sci_this_target[0].master_dark))
                
            if hasattr(sci_this_target[0], 'master_flat'):
                print('  Master Flat: ')
                print('    ' + str(sci_this_target[0].master_flat))

    @property
    def spec_module(self):
        return importlib.import_module(self.__module__)