# Python built in modules
from collections import OrderedDict
from abc import ABC, abstractmethod # Abstract classes
import glob # File searching
import copy
import os
import sys # sys utils
import pickle
import unicodedata
import pdb # debugging
stop = pdb.set_trace

# Science/math
import numpy as np # Math, Arrays
import sklearn # Clustering (DBSCAN)

# Graphics
import matplotlib.pyplot as plt

# LLVM
from numba import jit, njit
import numba

# Astropy
from astropy.time import Time
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as units

# User defined/pip modules
import pychell.maths as pcmath # mathy equations
import pychell.reduce.order_map as pcomap
import pychell.reduce.extract as pcextract


# Base class for a 2-dimensional echelle spectral image.
class SpecDataImage:
    """A base class for a spectral echelle image.

    Attributes:
        input_file (str): The full path + filename of the corresponding file.
        base_input_file (str): The filename of the corresponding file with the path removed.
        orientation (str): The orientation of the echelle orders on the detector. 'x' for aligned with rows, 'y' for columns. Defaults to 'x'.
    """
    
    def __init__(self, input_file, orientation='x'):
        """Default basic initializer.

        Args:
            input_file (str) The full path + filename of the corresponding file.
            orientation (str): The orientation of the echelle orders on the detector. 'x' for aligned with rows, 'y' for columns. Defaults to   'x'.
        """
        self.input_file = input_file
        self.base_input_file = os.path.basename(self.input_file)
        self.orientation = orientation
        
    
    # Given a list of SpecDataImage objects
    # this parses the images from their respective files and returns them as a cube
    @staticmethod
    def generate_data_cube(data_list):
        """Generates a data-cube (i.e., stack) of images.

        Args:
            data_list (list): A list of data objects.
        Returns:
            data_cube (np.ndarray): The generated data cube, with shape=(n_images, ny, nx).
        """
        n_data = len(data_list)
        data0 = data_list[0].parse_image()
        ny, nx = data0.shape
        data_cube = np.empty(shape=(n_data, ny, nx), dtype=float)
        data_cube[0, :, :] = data0
        for idata in range(1, n_data):
            data_cube[idata, :, :] = data_list[idata].parse_image()
            
        return data_cube
    
    # Parses the raw image given some info.
    def parse_image(self, hdu_num=0, correct_readmath=False):
        """Parses the input file for the image only. It's assumed the header has already been parsed.

        Args:
            hdu_num (int): Which header data unit the desired image is in, default to 0.
            correct_readmath (bool): Whether or not to correct the NDRS, BSCALE, and BZERO from the header.
        Returns:
            data_image (np.ndarray): The data image, with shape=(ny, nx)
        """
        # Parse the image
        fits_data = fits.open(self.input_file)[hdu_num]
        fits_data.verify('fix')
        data_image = fits_data.data.astype(float)
        
        # Correct silly things
        if correct_readmath:
            data_image = self.correct_readmath(data_image)
        if self.orientation != 'x':
            data_image = data_image.T
        return data_image
    
    def correct_readmath(self, data_image):
        """Corrects the NDRS, BSCALE, and BZERO. image_returned = (image_read_in - BZERO) / BSCALE

        Args:
            data_image (np.ndarray): The data image.
        Returns:
            data_image (np.ndarray): The corrected data image.
        """
        # Number of dynamic reads, or Non-destructive reads.
        # This reduces the read noise by sqrt(NDR)
        if hasattr(self, 'NDR'):
            data_image /= self.NDR
            
        # BZERO and BSCALE are common fits header keys for linear offsets
        if hasattr(self, 'BZERO'):
            data_image -= self.BZERO
        if hasattr(self, 'BSCALE'):
            data_image /= self.BSCALE

        return data_image
    
    def __str__(self):
        s = 'Spectral Image:' + '\n'
        s += '    Input File:' + self.base_input_file
        return s
    
    def __eq__(self, other):
        return self.input_file == other.input_file


    def __gt__(self, other):
        return self.time_of_obs.jd > other.time_of_obs.jd
        
    def __lt__(self, other):
        return self.time_of_obs.jd < other.time_of_obs.jd




#######################################
###### RAW (unmodified) DATA ##########
#######################################

# Class for a raw image (could be any science, calibration, etc.)
# Master calibration frames are separate.
class RawImage(SpecDataImage):
    """Base class for a raw, unmodified spectral data image (i.e., not modified other than possible readmath).

        Args:
            data_list (list): A list of data objects.
        Returns:
            data_cube (np.ndarray): The generated data cube, with shape=(n_images, ny, nx).
        """
    
    def __init__(self, input_file, orientation='x', header_keys=None, parse_header=False, output_dir_root=None, hdu_num=0, time_offset=0, filename_parser=None, img_num=None, n_tot_imgs=None):
        
        # Call super init
        super().__init__(input_file=input_file, orientation=orientation)
        
        if img_num is not None:
            self.img_num = img_num
            
        if n_tot_imgs is not None :
            self.n_tot_imgs = n_tot_imgs
        
        # Header info
        if header_keys is not None:
            self.header_keys = header_keys
        if parse_header and header_keys is not None:
            self.parse_header(hdu_num=hdu_num, time_offset=time_offset)
            
            
        # The image number and date of observation
        if filename_parser is not None:
            file_info = filename_parser(self.base_input_file)
            self.file_image_num = file_info['number']
            self.file_date_str = file_info['date']
            
        # Define output file directories if set and extract is true
        if output_dir_root is not None:
            
            # Determine a proper base string for filenames (ie, no fits or fz file extension)
            if self.base_input_file[-5:] == '.fits':
                base_str = self.base_input_file[:-4]
            elif self.base_input_file[-3:] == '.fz':
                base_str = self.base_input_file[:-2]
            else:
                base_str = self.base_input_file
                
            # The reduced spectra for all orders (flux, unc, badpix)
            self.out_file_spectrum = output_dir_root + 'spectra' + os.sep + base_str + self.target + str('_reduced') + '.fits'
            
            # The reduced spectra plots for all orders
            self.out_file_spectrum_plot = output_dir_root + 'previews' + os.sep + base_str + self.target + str('_reduced') + '.png'
            
            # The trace profiles for all orders (possibly oversampled)
            self.out_file_trace_profiles = output_dir_root + 'trace_profiles' + os.sep + base_str + self.target + str('_trace_profile') + '.npz'
            
            # The trace profiles for all orders (possibly oversampled)
            self.out_file_trace_profile_plots = output_dir_root + 'trace_profiles' + os.sep + base_str + self.target + str('_trace_profile') + '.png'
    
    # Load and store the header
    def parse_header(self, hdu_num=0, time_offset=0):
        
        # Parse the header
        fits_data = fits.open(self.input_file)[hdu_num]
        fits_data.verify('fix')
        self.header = fits_data.header
        
        # Store header keys
        for h in self.header_keys:
            if self.header_keys[h][0] in self.header:
                setattr(self, h, self.header[self.header_keys[h][0]])
            else:
                setattr(self, h, self.header_keys[h][1])
                
        # Store the sky coordinate
        self.coord = SkyCoord(ra=self.RA, dec=self.DEC, unit=(units.hourangle, units.deg))
        
        # Overwrite the time of observation
        self.time_of_obs = Time(float(self.time_of_obs) + time_offset, scale='utc', format='jd')
    
    # Outputs reduced orders for this spectrum
    def save_reduced_orders(self, data_arr, copy_keys=None, new_entries=None):
        header_out = copy.deepcopy(self.header)
        header_out.update(new_entries)
        if copy_keys is not None:
            for key in copy_keys:
                header_out[key] = getattr(key, self)
        hdu = fits.PrimaryHDU(data_arr, header=header_out)
        hdul = fits.HDUList([hdu])
        hdul.writeto(self.out_file_spectrum, overwrite=True, output_verify='ignore')
        
    def __str__(self):
        s = 'Raw Image:' + '\n'
        s += '    Input File:' + self.base_input_file
        return s
        

# Class for a a raw science frame
class ScienceImage(RawImage):
    
    def __init__(self, input_file=None, header_keys=None, parse_header=False, output_dir_root=None, hdu_num=0, time_offset=0, filename_parser=None, img_num=None, n_tot_imgs=None):
        
        # Call super init
        super().__init__(input_file=input_file, header_keys=header_keys, parse_header=parse_header, output_dir_root=output_dir_root, hdu_num=hdu_num, time_offset=time_offset, filename_parser=filename_parser, img_num=img_num, n_tot_imgs=n_tot_imgs)
    
    # Pairs based on the correct exposure time
    def pair_master_dark(self, master_darks):
        exp_times = np.array([master_darks[i].exp_time for i in range(len(master_darks))], dtype=float)
        matching_itime = np.where(self.exp_time == exp_times)[0]
        if matching_itime.size != 0:
            raise ValueError(str(matching_itime.size) + "master dark(s) found for\n" + str(self))
        else:
            self.master_dark = master_darks[matching_itime[0]]
    
    # Pairs based based on sky coords of the closest master flat and time between observation.
    # metric is d^2 = angular_sep^2 + time_diff^2
    def pair_master_flat(self, master_flats):
        ang_seps = np.array([np.abs(self.coord.separation(master_flats[i].coord)).value for i in range(len(master_flats))], dtype=float)
        time_seps = np.array([np.abs(self.coord.separation(master_flats[i].coord)).value for i in range(len(master_flats))], dtype=float)
        ds = np.sqrt(ang_seps**2 + time_seps**2)
        minds_loc = np.argmin(ds)
        self.master_flat = master_flats[minds_loc]
        
    def trace_orders(self, output_dir, extraction_settings, src=None):
        
        if src is None or src == 'empirical':
            order_dicts, order_map_image = pcomap.trace_orders_emprical(self.parse_image(), extraction_settings)
        elif src is 'from_flats':
            # Check if this flat has already been traced.
            if not self.master_flat.is_traced:
                order_dicts, order_map_image = pcomap.trace_orders_from_flat(self.master_flat.parse_image(), extraction_settings)
                self.master_flat.is_traced = True
                order_image_file = output_dir + 'order_map_' + self.file_date_str + '_' + self.target + '.fits'
                self.order_map = OrderMapImage('x', order_image_file, order_dicts)
                self.order_map.save(order_map_image=order_map_image)
            else:
                order_image_file = output_dir + 'order_map_' + self.file_date_str + '_' + self.target + '.fits'
                input_file_dicts = order_image_file[0:-4] + 'pkl'
                self.order_map = OrderMapImage.from_files(input_file_image=order_image_file, input_file_dicts=input_file_dicts)
        
    def __str__(self):
        s = 'Science Image:' + '\n'
        s += '    Input File: ' + self.base_input_file + '\n'
        s += '    Target: ' + self.target + '\n'
        s += '    UTC Date: ' + str(self.time_of_obs.datetime) + '\n'
        s += '    Slit: ' + self.slit + '\n'
        s += '    ' + unicodedata.lookup("GREEK SMALL LETTER LAMDA") + ': ' + self.wavelength_range + '\n'
        s += '    Gas Cell: ' + self.gas_cell + '\n'
        s += '    Exp. Time: ' + str(self.exp_time) + ' s' + '\n'
        
        if hasattr(self, 'master_bias'):
            s += '  ' + str(self.master_bias) + '\n'
            
        if hasattr(self, 'master_dark'):
            s += '  ' + str(self.master_dark) + '\n'
            
        if hasattr(self, 'master_flat'):
            s += '  ' + str(self.master_flat) + '\n'
        
        return s[:-1] # To remove the last new line character
    

# Class for raw bias frame 
class BiasImage(RawImage):
    
    def __init__(self, input_file=None, header_keys=None, parse_header=False, output_dir_root=None, hdu_num=0, time_offset=0, filename_parser=None, img_num=None, n_tot_imgs=None):
        
        # Call super init
        super().__init__(input_file=input_file, header_keys=header_keys, parse_header=parse_header, output_dir_root=output_dir_root, hdu_num=hdu_num, time_offset=time_offset, filename_parser=filename_parser, img_num=img_num, n_tot_imgs=n_tot_imgs)
        
    def __str__(self):
        s = 'Bias:' + '\n'
        s += '    Input File: ' + self.base_input_file
        return s

    
# Class for raw dark frame
class DarkImage(RawImage):
    
    def __init__(self, input_file=None, header_keys=None, parse_header=False, output_dir_root=None, hdu_num=0, time_offset=0, filename_parser=None, img_num=None, n_tot_imgs=None):
        
        # Call super init
        super().__init__(input_file=input_file, header_keys=header_keys, parse_header=parse_header, output_dir_root=output_dir_root, hdu_num=hdu_num, time_offset=time_offset, filename_parser=filename_parser, img_num=img_num, n_tot_imgs=n_tot_imgs)
        
    def __str__(self):
        s = 'Dark:' + '\n'
        s += '    Input File: ' + self.base_input_file
        s += '    Exp. Time: ' + self.exp_time
        return s
 
 
# Class for raw flat frame
class FlatImage(RawImage):
    
    def __init__(self, input_file=None, header_keys=None, parse_header=False, output_dir_root=None, hdu_num=0, time_offset=0, filename_parser=None, img_num=None, n_tot_imgs=None):
        
        # Call super init
        super().__init__(input_file=input_file, header_keys=header_keys, parse_header=parse_header, output_dir_root=output_dir_root, hdu_num=hdu_num, time_offset=time_offset, filename_parser=filename_parser, img_num=img_num, n_tot_imgs=n_tot_imgs)
        
    def __str__(self):
        s = 'Flat Field:' + '\n'
        s += '    Input File: ' + self.base_input_file
        s += '    Exp. Time: ' + self.exp_time
        return s
    


#######################################
###### Master Calibration Frames ######
#######################################

# Class for master calibration frame
# Actual calibration routines are in calib.py,
# So this is primarily a helpful container
class MasterCalibImage(SpecDataImage):
    
    def __init__(self, calib_images, input_file, orientation='x', header_keys=None):
        
        # Call super init
        super().__init__(input_file=input_file, orientation=orientation)
    
        # A list of the individual calibration image objects
        self.calib_images = calib_images
        
        # Store header keys initialize header
        self.header_keys = header_keys
        if self.header_keys is not None:
            self.generate_header()
        
    # Save the master calibration image
    def save(self, master_image, new_entries=None, input_file=None):
        
        if input_file is None:
            input_file = self.input_file
            
        header_out = copy.deepcopy(self.header)
        header_out.update(new_entries)
        hdu = fits.PrimaryHDU(master_image, header=header_out)
        hdul = fits.HDUList([hdu])
        hdul.writeto(self.input_file, overwrite=True, output_verify='ignore')
        
    def generate_header(self, header_keys=None, new_entries=None):
        
        # If header keys not already set
        if header_keys is not None:
            self.header_keys = header_keys
        
        # Copy the header from the first image
        self.header = copy.deepcopy(self.calib_images[0].header)
        
        # Store header keys
        for h in self.header_keys:
            if self.header_keys[h][0] in self.header:
                setattr(self, h, self.header[self.header_keys[h][0]])
            else:
                setattr(self, h, self.header_keys[h][1])
        
        # Compute the mean sky coords
        ras = [cdat.coord.ra.deg for cdat in self.calib_images]
        decs = [cdat.coord.dec.deg for cdat in self.calib_images]
        mra = np.average(ras)
        mdec = np.average(decs)
        self.coord = SkyCoord(ra=mra, dec=mdec, unit=(units.deg, units.deg))
        
        # Time of obs
        ts = np.average(np.array([cdat.time_of_obs.jd for cdat in self.calib_images], dtype=float))
        self.time_of_obs = Time(ts, scale='utc', format='jd')
        
    def __str__(self):
        s = 'Master Calibration Image:' + '\n'
        s += '    Input File: ' + self.base_input_file
        return s
    
    
class MasterFlatImage(MasterCalibImage):
    
    def __init__(self, calib_images, input_file, orientation='x', header_keys=None):
        
        # Super init
        super().__init__(calib_images=calib_images, input_file=input_file, orientation=orientation, header_keys=header_keys)
        
        self.is_traced = False
        
    # Pairs flats based on their location on the sky
    # flats is a list of all individual flat frame objects
    # Returns a list of MasterFlatImages
    @classmethod
    def from_all_flats(cls, flats, output_dir, header_keys=None):
    
        # Number of total individual flats
        n_flats = len(flats)
        
        # Create a clustering object
        density_cluster = sklearn.cluster.DBSCAN(eps=0.01745, min_samples=2, metric='euclidean', algorithm='auto', p=None, n_jobs=1)
        
        # Points are the ra and dec
        dist_matrix = np.empty(shape=(n_flats, n_flats), dtype=float)
        for i in range(n_flats):
            for j in range(n_flats):
                dpsi = np.abs(flats[i].coord.separation(flats[j].coord).value)
                dt = np.abs(flats[i].time_of_obs.jd - flats[j].time_of_obs.jd)
                dpsi /= np.pi
                dt /= 10  # Places more emphasis on delta psi
                dist_matrix[i, j] = np.sqrt(dpsi**2 + dt**2)
        
        # Fit
        db = density_cluster.fit(dist_matrix)
        
        # Extract the labels
        labels = db.labels_
        good_labels = np.where(labels >= 0)[0]
        if good_labels.size == 0:
            raise NameError('Error! The flat pairing algorithm failed. No usable labels found.')
        good_labels_init = labels[good_labels]
        labels_unique = np.unique(good_labels_init)
        
        # The number of master flats
        n_mflats = len(labels_unique)
        
        master_flats = []

        for l in range(n_mflats):
            
            # Get this flat label
            this_label = np.where(labels == labels_unique[l])[0]
            img_nums_this_label = np.array([int(flats[lb].file_image_num) for lb in this_label])
            img_start, img_end = str(np.min(img_nums_this_label)), str(np.max(img_nums_this_label))
            master_file = output_dir + os.sep + 'master_flat_' + flats[this_label[0]].file_date_str + '_imgs' + img_start + '-' + img_end + '_' + flats[this_label[0]].target + '.fits'
            master_flats.append(cls([flats[lb] for lb in this_label], master_file, header_keys=header_keys))
            
        return master_flats
        
        
    def __str__(self):
        s = 'Master Flat Field:' + '\n'
        s += '    Input File: ' + self.base_input_file + '\n'
        s += '    Exp. Time: ' + str(self.exp_time) + ' s'
        return s
    
    
class MasterDarkImage(MasterCalibImage):
    
    def __init__(self, calib_images, input_file, orientation='x', header_keys=None):
        
        # Super init
        super().__init__(calib_images=calib_images, input_file=input_file, orientation=orientation, header_keys=header_keys)
        
        # Nothing else done, so is redundant for now ...
        
        
    # Pairs darks based on their exposure times
    # darks is a list of all individual dark frame objects
    # Returns a list of MasterDarkImages
    @classmethod
    def from_all_darks(cls, calib_images, output_dir=None):
        
        # Exposure times
        exp_times_all = np.array([d.exp_time for d in calib_images], dtype=float)
        exp_times_unique = np.unique(exp_times_all)
        
        master_darks = []

        # Loop over each unique exposure time and identify the indices of darks that correspond to each other.
        for i in range(len(exp_times_unique)):
            inds = np.where(np.abs(exp_times_all - exp_times_unique[i]) < 0.1)[0]
            master_file = output_dir + 'master_dark_' + calib_images[0].file_date_str + str(exp_times_unique[i]) + 's' + '.fits'
            master_darks.append(cls([calib_images[i] for i in inds], master_file))
        
        return master_darks
        
    def __str__(self):
        s = 'Master Dark:' + '\n'
        s += '    Input File: ' + self.base_input_file + '\n'
        s += '    Exp. Time: ' + self.exp_time + ' s'
        return s
    
    
    
class MasterBiasImage(MasterCalibImage):
    
    def __init__(self, calib_images, input_file, orientation='x', header_keys=None):
        
        # Super init
        super().__init__(calib_images=calib_images, input_file=input_file, orientation=orientation, header_keys=header_keys)
        
        # Nothing else done, so is redundant for now ...
    
    def __str__(self):
        s = 'Master Bias:' + '\n'
        s += '    Input File: ' + self.base_input_file + '\n'
        return s
    
    
    
###########################
###### ORDER TRACING ######
###########################
    
    
class OrderMapImage(SpecDataImage):
    
    def __init__(self, orientation, input_file, order_dicts):
        
        # Call super init
        super().__init__(input_file=input_file, orientation=orientation)
        
        # Order dictionary info
        self.order_dicts = order_dicts
        self.input_file_dicts = self.input_file[0:-4] + 'pkl'
        
    
    @classmethod
    def from_files(cls, input_file_image, input_file_dicts):
        if input_file_dicts is not None:
            with open(input_file_dicts, 'rb') as handle:
                order_dicts = pickle.load(handle)
        return cls(orientation='x', input_file=input_file_image, order_dicts=order_dicts)
        
    def save(self, order_map_image=None):
        
        # Save the order map image
        if order_map_image is not None:
            hdu = fits.PrimaryHDU(order_map_image)
            hdul = fits.HDUList([hdu])
            hdul.writeto(self.input_file, overwrite=True)
            
        # Save the order map dictionaries
        with open(self.input_file_dicts, 'wb') as handle:
            pickle.dump(self.order_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def __str__(self):
        s = 'Order Map:' + '\n'
        s += '    Input File' + self.base_input_file
        return s
    
    
    
####################################################
###### MULTI TRACE DATA (under dev.) ###############
####################################################

#class MultiTraceImage():