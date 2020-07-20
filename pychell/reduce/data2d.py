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
import pychell.reduce.calib as pccalib
import pychell.spectrographs as spectrographs


# Base class for a 2-dimensional echelle spectral image.
class SpecDataImage:
    """A base class for a spectral echelle image.

    Attributes:
        input_file (str): The full path + filename of the corresponding file.
        base_input_file (str): The filename of the corresponding file with the path removed.
        parser (Parser): The parser object, must extend the Parser object.
        output_path (str): The root output path for this run.
    """
    
    def __init__(self, input_file, output_path):
        """Default basic initializer.

        Args:
            input_file (str):  The full path + filename of the corresponding file.
            output_path (str): The root output path for this run.
        """
        self.input_file = input_file
        self.base_input_file = os.path.basename(self.input_file)
        self.output_path = output_path
        
        # Determine image extension (if any) and remove
        if self.base_input_file.endswith('.fits'):
            self.input_file_noext = self.input_file[0:-5]
        elif self.base_input_file.endswith('.fz'):
            self.input_file_noext = self.input_file[0:-3]
        else:
            k = self.input_file.rfind('.')
            if k == -1:
                self.input_file_noext = self.base_input_file
            else:
                self.input_file_noext = self.base_input_file[0:k]
                
        self.base_input_file_noext = os.path.basename(self.input_file_noext)
        
    def parse_image(self):
        return fits.open(self.input_file)[0].data.astype(float)
        
    
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
    
    Attributes:
        parser (Parser): The parser object, must extend the Parser object.
    """
    
    def __init__(self, input_file, output_path, parser):
        
        # Call super init
        super().__init__(input_file=input_file, output_path=output_path)
        
        # Store the parser
        self.parser = parser
        
        # Parse the header and store.
        self.parser.parse_header(self)
        
        # Parse the image number
        self.parser.parse_image_num(self)
        
        # Parse the date of the observation
        self.parser.parse_date(self)
    
    
    # Outputs reduced orders for this spectrum
    def save_reduced_orders(self, reduced_orders):
        header_out = fits.Header()
        for key in self.header:
            try:
                header_out[key] = self.header[key]
            except:
                pass
        hdu = fits.PrimaryHDU(reduced_orders, header=header_out)
        hdul = fits.HDUList([hdu])
        fname = self.output_path + 'spectra' + os.sep + self.base_input_file_noext + '_' + self.target.replace(' ', '_') + '_reduced' + '.fits'
        hdul.writeto(fname, overwrite=True, output_verify='ignore')
        
    
    def parse_image(self):
        return self.parser.parse_image(self)    
    
    def __str__(self):
        s = 'Raw Image:' + '\n'
        s += '    Input File:' + self.base_input_file
        return s
        

# Class for a a raw science frame
class ScienceImage(RawImage):
    
    def __init__(self, input_file, output_path, parser):
        
        # Call super init
        super().__init__(input_file=input_file, output_path=output_path, parser=parser)
                
                
    def correct_flat_artifacts(self, output_dir, calibration_settings):
        
        # Check if this flat has already been corrected.
        if not self.master_flat.is_corrected:
            corrected_flat = pccalib.correct_flat_artifacts(self.master_flat, self.order_map, calibration_settings, output_dir)
            self.master_flat.is_corrected = True
            self.master_flat.save(corrected_flat)
        
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
    
    def __init__(self, input_file=None, header_keys=None, parse_header=False, output_path_root=None, hdu_num=0, time_offset=0, filename_parser=None, img_num=None, n_tot_imgs=None):
        
        # Call super init
        super().__init__(input_file=input_file, header_keys=header_keys, parse_header=parse_header, output_path_root=output_path_root, hdu_num=hdu_num, time_offset=time_offset, filename_parser=filename_parser, img_num=img_num, n_tot_imgs=n_tot_imgs)
        
    def __str__(self):
        s = 'Bias:' + '\n'
        s += '    Input File: ' + self.base_input_file
        return s

    
# Class for raw dark frame
class DarkImage(RawImage):
    
    def __init__(self, input_file=None, header_keys=None, parse_header=False, output_path_root=None, hdu_num=0, time_offset=0, filename_parser=None, img_num=None, n_tot_imgs=None):
        
        # Call super init
        super().__init__(input_file=input_file, header_keys=header_keys, parse_header=parse_header, output_path_root=output_path_root, hdu_num=hdu_num, time_offset=time_offset, filename_parser=filename_parser, img_num=img_num, n_tot_imgs=n_tot_imgs)
        
    def __str__(self):
        s = 'Dark:' + '\n'
        s += '    Input File: ' + self.base_input_file
        s += '    Exp. Time: ' + self.exp_time
        return s
 
 
# Class for raw flat frame
class FlatImage(RawImage):
    
    def __init__(self, input_file, output_path, parser):
        
        # Call super init
        super().__init__(input_file=input_file, output_path=output_path, parser=parser)
        
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
    
    def __init__(self, individuals):
        
        # Generate the filename for the median combined image
        img_nums = np.array([int(f.image_num) for f in individuals])
        img_start, img_end = str(np.min(img_nums)), str(np.max(img_nums))
        input_file = individuals[0].output_path + 'calib' + os.sep + 'master_flat_' + individuals[0].date_obs + '_imgs' + img_start + '-' + img_end + '.fits'
                
        # Call super init
        super().__init__(input_file=input_file, output_path=individuals[0].output_path)
    
        # A list of the individual calibration image objects
        self.individuals = individuals
        
        # Generate a header
        self.generate_header()
        
    # Save the master calibration image
    def save(self, master_image):
        header_out = fits.Header()
        for key in self.header:
            if type(self.header[key]) is str:
                header_out[key] = self.header[key]
        hdu = fits.PrimaryHDU(master_image, header=header_out)
        hdul = fits.HDUList([hdu])
        hdul.writeto(self.input_file, overwrite=True, output_verify='ignore')
        
    def generate_header(self):
        
        # Copy the header from the first image
        self.header = copy.deepcopy(self.individuals[0].header)
        
        # Store header keys
        for h in self.individuals[0].parser.header_keys:
            if self.individuals[0].parser.header_keys[h][0] in self.header:
                setattr(self, h, self.header[self.individuals[0].parser.header_keys[h][0]])
            else:
                setattr(self, h, self.individuals[0].parser.header_keys[h][1])
        
        # Compute the mean sky coords
        ras = [cdat.coord.ra.deg for cdat in self.individuals]
        decs = [cdat.coord.dec.deg for cdat in self.individuals]
        mra = np.average(ras)
        mdec = np.average(decs)
        self.header['coord'] = SkyCoord(ra=mra, dec=mdec, unit=(units.deg, units.deg))
        self.coord = self.header['coord']
        
        # Time of obs
        ts = np.average(np.array([cdat.time_of_obs.jd for cdat in self.individuals], dtype=float))
        self.header['time_of_obs'] = Time(ts, scale='utc', format='jd')
        self.time_of_obs = self.header['time_of_obs']
        
    def __str__(self):
        s = 'Master Calibration Image:' + '\n'
        s += '    Input File: ' + self.base_input_file
        return s
    
    
class MasterFlatImage(MasterCalibImage):
    
    def __init__(self, individuals):
        """
        Args:
            individuals (list of FlatFieldImages): The individual flat field images which generate this master flat field.
        """
        
        # Super init
        super().__init__(individuals=individuals)
        
        
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
    
# Order tracing is either:
# 1. Derived for each individual science exposure (empirical_unique)
# 2. Derived from a flat star and used for the whole night (empirical_flat_star)
# 3. Derived from flat fields and used for their corresponding images (empirical_flat_fields)
# 4. Hard coded (hard_coded)
# Each OrderMapImage corresponds to a fits file containing the labels / nans, and a dictionary with keys = labels (int), containing polynomial coffeicients (pcoeffs) and heights (height).
    
class OrderMapImage(SpecDataImage):
    
    def __init__(self, data, input_file):
        
        # Call super init
        super().__init__(input_file=input_file, output_path=data.output_path)
        
        # The data this map corresponds to
        self.data = data
        
        # Order dictionary file name
        self.input_file_orders_list = self.input_file_noext + '.pkl'
        
    def load(self):
        """Loads and returns the order map image and pickled dictionary file

        Returns:
            np.ndarray: The order map image
            list: The list of dictionaries for each order
        """
        with open(self.input_file_orders_list, 'rb') as f:
            orders_list = pickle.load(f)
        order_map_image = self.parse_image()
        if not hasattr(self, 'orders_list'):
            self.orders_list = orders_list
        return order_map_image, orders_list
            
    def __str__(self):
        s = 'Order Map:' + '\n'
        s += '    Input File' + self.base_input_file
        return s
    
    
class EmpiricalOrderMap(OrderMapImage):
    
    def __init__(self, data, method):
        
        # Generate the filename for the median combined image
        input_file = data.output_path + 'trace' + os.sep + data.base_input_file_noext + '_order_map.fits'
        
        # Call super init
        super().__init__(data=data, input_file=input_file)
        
        # Extract the method
        self.trace_alg = getattr(pcomap, method)
    
    
    def trace_orders(self, redux_settings):
        order_map_image, orders_list = self.trace_alg(self.data, redux_settings)
        self.orders_list = orders_list
        self.save(order_map_image)
        
    def save(self, order_map_image):
        
        # Save the order map image, default header
        hdu = fits.PrimaryHDU(order_map_image)
        hdul = fits.HDUList([hdu])
        hdul.writeto(self.input_file, overwrite=True)
            
        # Save the order map dictionaries
        with open(self.input_file_orders_list, 'wb') as f:
            pickle.dump(self.orders_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    def __str__(self):
        s = 'Empirical Order Map:' + '\n'
        s += '    Input File' + self.base_input_file
        return s
    
    
class HardCodedOrderMap(OrderMapImage):
    
    def __init__(self, data):
        
        # Input file is stored as a class variable
        input_file = self.input_file
        
        # Call super init
        super().__init__(data, input_file=input_file)
            
    def __str__(self):
        s = 'Hard Coded Order Map:' + '\n'
        s += '    Input File' + self.base_input_file
        return s

class iSHELLKGasMap(HardCodedOrderMap):
    
    input_file = os.path.dirname(spectrographs.__file__) + os.sep + 'ishell' + os.sep + 'order_map_KGAS.fits'
    
    def __init__(self, data):
        
        super().__init__(data=data)
        
        
class CHIRONMap(HardCodedOrderMap):
    
    input_file = os.path.dirname(spectrographs.__file__) + os.sep + 'chiron' + os.sep + 'chiron_order_map_master2.fits'
    
    def __init__(self, data):
        
        super().__init__(data=data)
        
        
class FlatFieldOrderMap(EmpiricalOrderMap):
    
    def __init__(self, data, method=None):
        
        if method is None:
            method = 'trace_orders_from_flat_field'
        
        super().__init__(data=data, method=method)
    
    
class FlatStarOrderMap(EmpiricalOrderMap):
    
    def __init__(self, data, input_file, output_path, method=None):
        
        if method is None:
            method = 'trace_orders_empirical'
        
        super().__init__(data=data, input_file=input_file, output_path=output_path, method=method)
        
        
class ScienceOrderMap(EmpiricalOrderMap):
    
    def __init__(self, data, method=None):
        
        if method is None:
            method = 'trace_orders_empirical'
        
        super().__init__(data=data, method=method)

############################
###### Co-added Image ######
############################

class CoAddedImage(SpecDataImage):
    
    def __init__(self, input_file, base_images, orientation='x'):
        
        super().__init__(input_file, orientation=orientation)
        
        # Stores the base images
        self.base_images = base_images
        
    def coadd(self, method='sum'):
        """Co-adds the images
        
        Kwargs:
            method (str): Either sum or median.
        """
        data_cube = self.generate_data_cube(base_images)
        if method == 'sum':
            return np.nansum(data_cube, axis=0)
        elif method == 'median':
            return np.nanmedian(data_cube, axis=0)
        return data_cube
        
        

#####################################
#### Instrument Specific Parsing ####
#####################################


#### Each function must be named parse_insname (all lowercase) to be automatically recognized
#### The signature of these functions must take an input path, and return a dictionary of images where each key is a type of above image.

class Parser:
    
    def __init__(self):
        # nothing is Done.
        pass
    
    # Defaults to parse method
    def __call__(self, redux_settings):
        return self.categorize(redux_settings)
    
    def categorize(self, redux_settings):
        raise NotImplementedError("Must implement a categorize method for this instrument")
    
    def parse_header(self, data):
        
        # Stores extracted key, value pairs
        modified_header = {}
        
        # Parse the header
        fits_data = fits.open(data.input_file)[0]
        
        # Just in case
        try:
            fits_data.verify('fix')
        except:
            pass

        # The current header
        current_fits_header = fits_data.header
        modified_header.update(current_fits_header)
        
        # Store header keys as consistent names
        for h in self.header_keys:
            if self.header_keys[h][0] in current_fits_header:
                modified_header[h] = current_fits_header[self.header_keys[h][0]]
            else:
                modified_header[h] = self.header_keys[h][1]
                
        # Also pass important keys as data object attributes
        for key in self.header_keys:
            if not hasattr(data, key):
                setattr(data, key, modified_header[key])
        return modified_header
    
    def parse_image(self, filename):
        """Parses the input file for the image only. It's assumed the header has already been parsed.

        Args:
            filename (str): The filename.
        Returns:
            data_image (np.ndarray): The data image, with shape=(ny, nx)
        """
        return fits.open(filename)[0].data.astype(float)
    
    def parse_date(self, data):
        raise NotImplementedError("Must implement a parse_date method for this instrument")
    
    def parse_image_num(self, data):
        raise NotImplementedError("Must implement a parse_image_num method for this instrument")
    
    # Functions that may be overloaded
    def correct_readmath(self, data, data_image):
        """Corrects the NDRS, BSCALE, and BZERO in place. image_corrected = (image_original / NDR - BZERO) / BSCALE

        Args:
            data (SpecDataImage): The SpecDataImage object this image corresponds to
            data_image (np.ndarray): The data image.
        """
        # Number of dynamic reads, or Non-destructive reads.
        # This reduces the read noise by sqrt(NDR)
        if hasattr(data, 'NDR'):
            data_image /= float(data.NDR)
            
        # BZERO and BSCALE are common fits header keys for linear transformations
        if hasattr(data, 'BZERO'):
            data_image -= float(data.BZERO)
        if hasattr(data, 'BSCALE'):
            data_image /= float(data.BSCALE)
            
    def group_darks(self, darks):
        master_darks = []
        exp_times = np.array([d.exp_time for d in darks])
        exp_times_unq = np.unique(exp_times)
        for t in exp_times_unq:
            good = np.where(exp_times == t)[0]
            indiv_darks = [darks[i] for i in good]
            master_darks.append(MasterDarkImage(indiv_darks))
        return master_darks
    
    def group_flats(self, flats):
        
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
            this_label = np.where(good_labels_init == labels_unique[l])[0]
            indiv_flats = [flats[lb] for lb in this_label]
            master_flats.append(MasterFlatImage(indiv_flats))
            
        return master_flats
    
    # Pairs Master bias
    def pair_master_bias(self, spec_image, master_bias):
        spec_image.master_bias = master_bias
    
    # Pairs based on the correct exposure time
    def pair_master_dark(self, spec_image, master_darks):
        exp_times = np.array([master_darks[i].exp_time for i in range(len(master_darks))], dtype=float)
        good = np.where(spec_image.exp_time == exp_times)[0]
        if good.size != 0:
            raise ValueError(str(good.size) + "master dark(s) found for\n" + str(self))
        else:
            spec_image.master_dark = master_darks[good[0]]
    
    # Pairs based based on sky coords of the closest master flat and time between observation.
    # metric is d^2 = angular_sep^2 + time_diff^2
    def pair_master_flat(self, spec_image, master_flats):
        ang_seps = np.array([np.abs(spec_image.coord.separation(mf.coord)).value for mf in master_flats], dtype=float)
        time_seps = np.array([np.abs(spec_image.time_of_obs.value - mf.time_of_obs.value) for mf in master_flats], dtype=float)
        ds = np.sqrt(ang_seps**2 + time_seps**2)
        minds_loc = np.argmin(ds)
        spec_image.master_flat = master_flats[minds_loc]
        
    def print_summary(self, data):
    
        n_sci_tot = len(data['science'])
        targets_all = np.array([data['science'][i].target for i in range(n_sci_tot)], dtype='<U50')
        targets_unique = np.unique(targets_all)
        for i in range(len(targets_unique)):
            
            target = targets_unique[i]
            
            locs_this_target = np.where(targets_all == target)[0]
            
            sci_this_target = [data['science'][j] for j in locs_this_target]
            
            print('Target: ' + target)
            print('    N Exposures: ' + str(locs_this_target.size))
            if hasattr(sci_this_target[0], 'master_bias'):
                print('    Master Bias File(s): ')
                print('    ' + data['science'].master_bias.base_input_file)
                
            if hasattr(sci_this_target[0], 'master_dark'):
                darks_this_target_all = np.array([sci.master_dark for sci in sci_this_target], dtype=DarkImage)
                darks_unique = np.unique(darks_this_target_all)
                print('  Master Dark File(s): ')
                for d in darks_unique:
                    print('    ' + d.base_input_file)
                
            if hasattr(sci_this_target[0], 'master_flat'):
                flats_this_target_all = np.array([sci.master_flat for sci in sci_this_target], dtype=FlatImage)
                flats_unique = np.unique(flats_this_target_all)
                print('  Master Flat File(s): ')
                for f in flats_unique:
                    print('    ' + f.base_input_file)
                    
            print('')
        
    
    
class iSHELLParser(Parser):
    
    header_keys = {
        'target': ['OBJECT', ValueError],
        'RA': ['TCS_RA', ValueError],
        'DEC': ['TCS_DEC', ValueError],
        'slit': ['SLIT', ValueError],
        'wavelength_range': ['XDTILT', None],
        'gas_cell': ['GASCELL', None],
        'exp_time': ['ITIME', ValueError],
        'time_of_obs': ['TCS_UTC', ValueError],
        'NDR': ['NDR', 1],
        'BZERO': ['BZERO', 0],
        'BSCALE': ['BSCALE', 1]
    }
        
    def categorize(self, redux_settings):
        
        # Stores the data as above objects
        data = {}
        
        # iSHELL science files are files that contain spc or data
        sci_files1 = glob.glob(redux_settings['input_path'] + '*data*.fits')
        sci_files2 = glob.glob(redux_settings['input_path'] + '*spc*.fits')
        sci_files = sci_files1 + sci_files2
        sci_files = np.sort(np.unique(np.array(sci_files, dtype='<U200'))).tolist()
        
        data['science'] = [ScienceImage(input_file=sci_files[f], output_path=redux_settings['run_output_path'], parser=self) for f in range(len(sci_files))]
    
        # Bias (typically not done for iSHELL, but still check)
        if redux_settings['bias_subtraction']:
            bias_files = glob.glob(redux_settings['input_path'] + '*bias*.fits')
            data['bias'] = [BiasImage(input_file=bias_files[f], output_path=redux_settings['run_output_path'], parser=self) for f in range(len(bias_files))]
            data['master_bias'] = MasterBiasImage(individuals=data['bias'], output_path=redux_settings['run_output_path'])
            
            for sci in data['science']:
                self.pair_master_bias(sci, data['master_bias'])
                
            for flat in data['flats']:
                self.pair_master_bias(flat, data['master_bias'])
            
        # Darks assumed to contain dark in filename
        if redux_settings['dark_subtraction']:
            dark_files = glob.glob(redux_settings['input_path'] + '*dark*.fits')
            data['darks'] = [DarkImage(input_file=dark_files[f], output_path=redux_settings['run_output_path'], parser=self) for f in range(len(dark_files))]
            data['master_darks'] = MasterDarkImage.from_all_darks(data['darks'])
            
            for sci in data['science']:
                self.pair_master_dark(sci, data['master_darks'])
                
            for flat in data['flats']:
                self.pair_master_dark(flat, data['master_darks'])
        
        # iSHELL flats must contain flat in the filename
        if redux_settings['flat_division']:
            flat_files = glob.glob(redux_settings['input_path'] + '*flat*.fits')
            data['flats'] = [FlatImage(input_file=flat_files[f], output_path=redux_settings['run_output_path'], parser=self) for f in range(len(flat_files))]
            data['master_flats'] = self.group_flats(data['flats'])
            for sci in data['science']:
                self.pair_master_flat(sci, data['master_flats'])
            
        # iSHELL ThAr images must contain arc (not implemented yet)
        if redux_settings['wavelength_calibration']:
            thar_files = glob.glob(redux_settings['input_path'] + '*arc*.fits')
            data['wavecals'] = [ThArImage(input_file=thar_files[f], output_path=redux_settings['run_output_path'] + 'calib' + os.sep, parser=self) for f in range(len(thar_files))]
            data['master_wavecals'] = self.group_wavecals(data['wavecals'])
            for sci in data['science']:
                self.pair_master_wavecal(sci, data['master_wavecals'])
                
        
        self.print_summary(data)

        return data

    # icm.2019A076.190627.flat.00019.a.fits
    def parse_image_num(self, data):
        string_list = data.base_input_file.split('.')
        data.image_num = string_list[4]
        
    def parse_image(self, data):
        """Parses the input file for the image only. It's assumed the header has already been parsed.

        Args:
            hdu_num (int): Which header data unit the desired image is in, default to 0.
        Returns:
            data_image (np.ndarray): The data image, with shape=(ny, nx)
        """
        # Parse the image
        data_image = super().parse_image(data.input_file)
        
        # Correct readmath (checks if present)
        self.correct_readmath(data, data_image)
        return data_image
        
    # icm.2019A076.190627.flat.00019.a.fits
    def parse_date(self, data):
        string_list = data.base_input_file.split('.')
        data.date_obs = string_list[1][0:4] + string_list[2][2:]
    
    def parse_header(self, data):
        
        modified_header = super().parse_header(data)
                
        # Store the sky coordinate
        modified_header['coord'] = SkyCoord(ra=modified_header['RA'], dec=modified_header['DEC'], unit=(units.hourangle, units.deg))
        data.coord = modified_header['coord']

        # Overwrite the time of observation
        modified_header['time_of_obs'] = Time(float(modified_header['time_of_obs']) + 2400000.5, scale='utc', format='jd')
        data.time_of_obs = modified_header['time_of_obs']
        
        data.header = modified_header
        
class NIRSPECParser(Parser):
    
    header_keys = {
        'target': ['OBJECT', ValueError],
        'RA': ['RA', ValueError],
        'DEC': ['DEC', ValueError],
        'slit': ['SLITWIDT', ValueError],
        'wavelength_range': ['_NO_KEY_', None],
        'gas_cell': ['_NO_KEY_', None],
        'exp_time': ['ITIME', ValueError],
        'time_of_obs': ['MJD-OBS', ValueError],
        'NDR': ['NDR', 1],
        'BZERO': ['BZERO', 0],
        'BSCALE': ['BSCALE', 1]
    }
        
    def categorize(self, redux_settings):
        
        # Stores the data as above objects
        data = {}
        
        # NIRSPEC science files
        sci_files = glob.glob(redux_settings['input_path'] + '*.fits')
        sci_files = np.sort(np.unique(np.array(sci_files, dtype='<U200'))).tolist()
        
        data['science'] = [ScienceImage(input_file=sci_files[f], output_path=redux_settings['run_output_path'], parser=self) for f in range(len(sci_files))]
    
        # Bias
        if redux_settings['bias_subtraction']:
            bias_files = glob.glob(redux_settings['input_path'] + '*bias*.fits')
            data['bias'] = [BiasImage(input_file=bias_files[f], output_path=redux_settings['run_output_path'], parser=self) for f in range(len(bias_files))]
            data['master_bias'] = MasterBiasImage(individuals=data['bias'], output_path=redux_settings['run_output_path'])
            
            for sci in data['science']:
                self.pair_master_bias(sci, data['master_bias'])
                
            for flat in data['flats']:
                self.pair_master_bias(flat, data['master_bias'])
            
        # Darks assumed to contain dark in filename
        if redux_settings['dark_subtraction']:
            dark_files = glob.glob(redux_settings['input_path'] + '*dark*.fits')
            data['darks'] = [DarkImage(input_file=dark_files[f], output_path=redux_settings['run_output_path'], parser=self) for f in range(len(dark_files))]
            data['master_darks'] = MasterDarkImage.from_all_darks(data['darks'])
            
            for sci in data['science']:
                self.pair_master_dark(sci, data['master_darks'])
                
            for flat in data['flats']:
                self.pair_master_dark(flat, data['master_darks'])
        
        # Flats must contain flat in the filename
        if redux_settings['flat_division']:
            flat_files = glob.glob(redux_settings['input_path'] + '*flat*.fits')
            data['flats'] = [FlatImage(input_file=flat_files[f], output_path=redux_settings['run_output_path'], parser=self) for f in range(len(flat_files))]
            data['master_flats'] = self.group_flats(data['flats'])
            for sci in data['science']:
                self.pair_master_flat(sci, data['master_flats'])
        
        self.print_summary(data)

        return data

    # NS.20050601.48268.fits
    def parse_image_num(self, data):
        string_list = data.base_input_file.split('.')
        data.image_num = string_list[2]
        
    def parse_image(self, data):
        """Parses the input file for the image only. It's assumed the header has already been parsed.

        Args:
            hdu_num (int): Which header data unit the desired image is in, default to 0.
        Returns:
            data_image (np.ndarray): The data image, with shape=(ny, nx)
        """
        # Parse the image
        data_image = super().parse_image(data.input_file)
        
        # Correct readmath (checks if present)
        self.correct_readmath(data, data_image)
        return data_image
        
    # NS.20050601.48268.fits
    def parse_date(self, data):
        string_list = data.base_input_file.split('.')
        data.date_obs = string_list[1]
    
    def parse_header(self, data):
        
        modified_header = super().parse_header(data)
                
        # Store the sky coordinate
        modified_header['coord'] = SkyCoord(ra=modified_header['RA'], dec=modified_header['DEC'], unit=(units.hourangle, units.deg))
        data.coord = modified_header['coord']

        # Overwrite the time of observation
        modified_header['time_of_obs'] = Time(float(modified_header['time_of_obs']) + 2400000.5, scale='utc', format='jd')
        data.time_of_obs = modified_header['time_of_obs']
        
        data.header = modified_header

# Under dev
class CHIRONParser(Parser):
    
    header_keys = {
        'target': ['OBJECT', ValueError],
        'RA': ['RA', ValueError],
        'DEC': ['DEC', ValueError],
        'slit': ['DECKER', ValueError],
        'wavelength_range': ['_NOKEY_', 'Fixed (415 nm - 880 nm)'],
        'exp_time': ['ITIME', ValueError],
        'gas_cell': ['IODCELL', 'Out'],
        'time_of_obs': ['DATE-OBS', ValueError],
        'NDR': ['NDR', 1],
        'BZERO': ['BZERO', 0],
        'BSCALE': ['BSCALE', 1]
    }
        
    def categorize(self, redux_settings):
        
        # Stores the data as above objects
        data = {}
        
        # iSHELL science files are files that contain spc or data
        sci_files = glob.glob(redux_settings['input_path'] + '*.fits')
        sci_files = np.sort(np.unique(np.array(sci_files, dtype='<U200'))).tolist()
        
        data['science'] = [ScienceImage(input_file=sci_files[f], output_path=redux_settings['run_output_path'], parser=self) for f in range(len(sci_files))]
    
        # Bias
        if redux_settings['bias_subtraction']:
            bias_files = glob.glob(redux_settings['input_path'] + '*bias*.fits')
            data['bias'] = [BiasImage(input_file=bias_files[f], output_path=redux_settings['run_output_path'], parser=self) for f in range(len(bias_files))]
            data['master_bias'] = MasterBiasImage(individuals=data['bias'], output_path=redux_settings['run_output_path'])
            
        # Darks assumed to contain dark in filename
        if redux_settings['dark_subtraction']:
            dark_files = glob.glob(redux_settings['input_path'] + '*dark*.fits')
            data['darks'] = [DarkImage(input_file=dark_files[f], output_path=redux_settings['run_output_path'], parser=self) for f in range(len(dark_files))]
            data['master_darks'] = MasterDarkImage.from_all_darks(data['darks'])
        
        # CHIRON flats must contain flat in the filename
        if redux_settings['flat_division']:
            flat_files = glob.glob(redux_settings['input_path'] + '*flat*.fits')
            data['flats'] = [FlatImage(input_file=flat_files[f], output_path=redux_settings['run_output_path'], parser=self) for f in range(len(flat_files))]
            data['master_flats'] = self.group_flats(data['flats'])
            for sci in data['science']:
                self.pair_master_flat(sci, data['master_flats'])
            
        # CHIRON ThAr images must contain arc
        if redux_settings['wavelength_calibration']:
            thar_files = glob.glob(redux_settings['input_path'] + '*arc*.fits')
            data['wave_cals'] = [ThArImage(input_file=thar_files[f], output_path=redux_settings['run_output_path'] + 'calib' + os.sep, parser=self) for f in range(len(thar_files))]
            
        self.print_summary(data)

        return data

    # chi190915.1242.fits
    def parse_image_num(self, data):
        string_list = data.base_input_file.split('.')
        data.image_num = string_list[1]
        
    def parse_image(self, data):
        """Parses the input file for the image only. It's assumed the header has already been parsed.

        Args:
            hdu_num (int): Which header data unit the desired image is in, default to 0.
        Returns:
            data_image (np.ndarray): The data image, with shape=(ny, nx)
        """
        # Parse the image
        return super().parse_image(data.input_file).T
        
    # chi190915.1242.fits
    def parse_date(self, data):
        string_list = data.base_input_file.split('.')
        data.date_obs = '20' + string_list[0][3:]
    
    def parse_header(self, data):
        
        modified_header = super().parse_header(data)
                
        # Store the sky coordinate
        modified_header['coord'] = SkyCoord(ra=modified_header['RA'], dec=modified_header['DEC'], unit=(units.hourangle, units.deg))
        data.coord = modified_header['coord']

        # Overwrite the time of observation
        tt = modified_header['time_of_obs'].replace('T', '-').replace(':', '-').split('-')
        modified_header['time_of_obs'] = Time({'year': int(tt[0]), 'month': int(tt[1]), 'day': int(tt[2]), 'hour': int(tt[3]), 'minute':  int(tt[4]),'second': float(tt[5])})
        data.time_of_obs = modified_header['time_of_obs']
        
        data.header = modified_header