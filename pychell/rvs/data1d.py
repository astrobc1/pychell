# Python built in modules
from collections import OrderedDict
from abc import ABC, abstractmethod # Abstract classes
import glob # File searching
import os # OS 
import sys # sys utils
from pdb import set_trace as stop # debugging

# Science/math
import numpy as np # Math, Arrays

# Graphics
import matplotlib.pyplot as plt

# LLVM
from numba import jit, njit
import numba

# Astropy
from astropy.time import Time
from astropy.io import fits

# User defined/pip modules
import pychell.maths as pcmath # mathy equations
import pychell.rvs.template_augmenter as pcaugmenter

# Base class for a reduced 1-dimensional spectrum
class SpecData1d:
    """ Base class for an extracted 1-dimensional spectrum.

    Attributes:
        input_file (str): The full path + input file this spectrum corresponds to, possibly containing all orders.
        base_input_file (str): The filename this spectrum corresponds to, with the path removed.
        order_num (int): The image order number, defaults to None.
        spec_num (int): The spectral image number, defaults to None.
        flux (np.ndarray): The normalized flux.
        flux_unc (np.ndarray): The normalized flux uncertainty.
        badpix (np.ndarray): The badpix array (1=good, 0=bad)
        wave_grid (np.ndarray): If present, the known (or possibly only an initial) wavelength grid of the data.
    """

    
    def __init__(self, input_file, forward_model):
        """ Base initialization for this model component.

        Args:
            input_file (str): The full path + input file this spectrum corresponds to, possibly containing all orders.
            order_num (int): The image order number, defaults to None.
            spec_num (int): The spectral image number, defaults to None.
            crop_pix (list): Pixels to crop on the left and right of the data arrays. Pixels are not removed, but rather changed to nan with corresponding values of zero in the bad pixel mask, defaults to None, or no cropped pixels. If pixels are already cropped, then this will still be performed but have no effect, which is fine.
        """
        # Store the input file, spec, and order num
        self.input_file = input_file
        self.base_input_file = os.path.basename(self.input_file)
        
        # Order number and image number if set
        self.order_num = forward_model.order_num
        self.spec_num = forward_model.spec_num
        self.crop_pix = forward_model.crop_data_pix
            
        # Default wavelength and LSF grid, may be overwritten in custom parse method.
        self.default_wave_grid = None
        self.default_lsf = None
        
        # Parse the data for this observation
        self.parse(forward_model)
        
        # Enforce the pixels are cropped (ideally they are already cropped and this has no effect, but still optional)
        if self.crop_pix is not None:
            
            # Flux
            self.flux[0:self.crop_pix[0]] = np.nan
            self.flux[self.flux.size - self.crop_pix[1] - 1:] = np.nan
            
            # Flux unc
            self.flux_unc[0:self.crop_pix[0]] = np.nan
            self.flux_unc[self.flux.size - self.crop_pix[1] - 1:] = np.nan
            
            # Bad pix
            self.badpix[0:self.crop_pix[0]] = 0
            self.badpix[self.flux.size - self.crop_pix[1] - 1:] = 0
            
        # Mask negative flux
        bad = np.where(self.flux <= 0)[0]
        if bad.size > 0:
            self.flux[bad] = np.nan
            self.flux_unc[bad] = np.nan
            self.badpix[bad] = 0
            
        # Further mask very low flux
        bad = np.where(self.flux < 0.05)[0]
        if bad.size > 0:
            self.flux[bad] = np.nan
            self.flux_unc[bad] = np.nan
            self.flux[bad] = 0
        
    # Calculate bc info for only this observation
    def calculate_bc_info(self, obs_name, star_name):
        """ Computes the bary-center Julian Day and bary-center velocities for this observation and stores them in place.

        Args:
            obs_name (str): The name of the observatory to be looked up on EarthLocations.
            star_name (str): The name of the star to be queuried on SIMBAD.
        """
        
        # Import the barycorrpy module
        from barycorrpy import get_BC_vel
        from barycorrpy.utc_tdb import JDUTC_to_BJDTDB
        
        # BJDs
        self.bjd = JDUTC_to_BJDTDB(JDUTC=self.JD, starname=star_name, obsname=obs_name)[0]
        
        # bc vels
        self.bc_vel = get_BC_vel(JDUTC=self.JD, starname=star_name, obsname=obs_name)[0]
    
    def set_bc_info(self, bjd=None, bc_vel=None):
        """ Basic setter method for the BJD and barycenter velocity.

        Args:
            bjd (float): The BJD of the observation.
            bc_vel (float): The bary-center velocity of the observation.
        """
        if bjd is not None:
            self.BJD = bjd
        if bc_vel is not None:
            self.bc_vel = bc_vel
        

    def parse(self, *args, **kwargs):
        """ A default parse method which must be implemented for each instrument.
        """
        raise NotImplementedError("Must implement a parse() routine for this instrument!")
    
    
    @staticmethod
    def load_custom_bcinfo(file):
        """ Loads a custom bc file.

        Args:
            file (str): The full path + filename of the file containing the information. The file is assumed to be comma separated, with col1=bjds, col2=vels.
        Returns:
            bjds (np.ndarray): The BJDs of the observations
            bc_vels (np.ndarray): The bary-center velocities of the observations.
        """
        bjds, bc_vels = np.loadtxt(file, delimiter=',')
        return bjds, bc_vels
    
    
    @staticmethod
    def calculate_bc_info_all(forward_models, obs_name, star_name):
        """ Computes the bary-center information for all observations.

        Args:
            forward_models (ForwardModels): The list of forward model objects.
            obs_name (str): The name of the observatory to be looked up on EarthLocations.
            star_name (str): The name of the star to be queuried on SIMBAD.
        Returns:
            bjds (np.ndarray): The BJDs of the observations
            bc_vels (np.ndarray): The bary-center velocities of the observations.
        """
        # Import the barycorrpy module
        from barycorrpy import get_BC_vel
        from barycorrpy.utc_tdb import JDUTC_to_BJDTDB
        
        # Extract the jds
        jds = np.array([fwm.data.JD for fwm in forward_models], dtype=float)
        
        # BJDs
        bjds = JDUTC_to_BJDTDB(JDUTC=jds, starname=star_name, obsname=obs_name)[0]
        
        # bc vels
        bc_vels = get_BC_vel(JDUTC=jds, starname=star_name, obsname=obs_name)[0]
        
        for i in range(len(forward_models)):
            forward_models[i].data.set_bc_info(bjd=bjds[i], bc_vel=bc_vels[i])
            
        return bjds, bc_vels

    
class SpecDataiSHELL(SpecData1d):
    """ Class for extracted 1-dimensional spectra from iSHELL on the NASA IRTF.
    """
        
    def parse(self, forward_model):
        """ Parses iSHELL data and computes the mid-exposure time (no exp meter for iSHELL).
        """
        
        # Load the flux, flux unc, and bad pix arrays
        fits_data = fits.open(self.input_file)[0]
        fits_data.verify('fix')
        oi = self.order_num - 1
        self.flux, self.flux_unc, self.badpix = fits_data.data[oi, 0, :, 0].astype(np.float64), fits_data.data[oi, 0, :, 1].astype(np.float64), fits_data.data[oi, 0, :, 2].astype(np.float64)

        # Flip the data so wavelength is increasing for iSHELL data
        self.flux = self.flux[::-1]
        self.badpix = self.badpix[::-1]
        self.flux_unc = self.flux_unc[::-1]
        
        # Normalize to 99th percentile
        med_val = pcmath.weighted_median(self.flux, med_val=0.99)
        self.flux /= med_val
        self.flux_unc /= med_val
        
        # Define the JD of this observation using the mid point of the observation
        self.JD = float(fits_data.header['TCS_UTC']) + 2400000.5 + float(fits_data.header['ITIME']) / (2 * 3600 * 24)
        

class SpecDataCHIRON(SpecData1d):
    """ Class for extracted 1-dimensional spectra from CHIRON on the SMARTS 1.5 m telescope.
    """
    def parse(self, forward_model):
        """ Parses CHIRON data and extracts the flux weighted midpoint of the exposure from the header if present, otherwise computes the mid exposure time. The flux uncertainty is not provided from CHIRON (?), so we assume all normalized uncertainties to arbitrarily be 0.001 (uniform). The wavelength grid provided by the ThAr lamp is provided in the wave_grid attribute.
        """
        
        # Load the flux, flux unc, and bad pix arrays
        fits_data = fits.open(self.input_file)[0]
        fits_data.verify('fix')
        
        self.default_wave_grid, self.flux = fits_data.data[self.order_num - 1, :, 0].astype(np.float64), fits_data.data[self.order_num - 1, :, 1].astype(np.float64)
        self.flux /= pcmath.weighted_median(self.flux, med_val=0.98)
        
        # For CHIRON, generate a dumby uncertainty grid and a bad pix array that will be updated or used
        self.flux_unc = np.zeros_like(self.flux) + 1E-3
        self.badpix = np.ones_like(self.flux)
        
        # JD from exposure meter. Sometimes it is not set in the header, so use the timing mid point in that case.
        if not (fits_data.header['EMMNWB'][0:2] == '00'):
            self.JD = Time(fits_data.header['EMMNWB'].replace('T', ' '), scale='utc').jd
        else:
            self.JD = Time(fits_data.header['DATE-OBS'].replace('T', ' '), scale='utc').jd + float(fits_data.header['EXPTIME']) / (2 * 3600 * 24)
        
        
class SpecDataPARVI(SpecData1d):
    
    """ Class for extracted 1-dimensional spectra from PARVI.
    """    
    def parse(self, forward_model):
        
        # Load the flux, flux unc, and bad pix arrays. Also load the known wavelength grid for a starting point
        fits_data = fits.open(self.input_file)[0]
        oi = self.order_num - 1
        self.wave_grid, self.flux, self.flux_unc = fits_data.data[oi, :, 0].astype(np.float64), fits_data.data[oi, :, 5].astype(np.float64), fits_data.data[oi, :, 6].astype(np.float64)
        
        # Normalize according to 98th percentile in flux
        continuum = pcmath.weighted_median(self.flux, med_val=0.98)
        self.flux /= continuum
        self.flux_unc /= continuum
        
        # Create bad pix array, further cropped later
        self.badpix = np.ones(self.flux.size, dtype=np.float64)
        bad = np.where(~np.isfinite(self.flux) | (self.flux == 0) | (self.flux_unc == 0) | (self.flux_unc > 0.5))[0]
        if bad.size > 0:
            self.badpix[bad] = 0
        
        # Convert wavelength grid to Angstroms, required!
        self.wave_grid *= 10
        self.JD = float(fits_data.header['JD'])
        
        
class SpecDataMinervaAustralis(SpecData1d):
        
    def parse(self, forward_model):
        
        # Load the flux, flux unc, and bad pix arrays
        # TOI257_ThAr_KiwiSpec_2019Aug05_0007_wcal_fib3
        self.default_wave_grid = np.loadtxt(self.input_file + '_wave.txt').T[:, ::-1][:, self.order_num]
        self.flux = np.loadtxt(self.input_file + '_spec.txt').T[:, ::-1][:, self.order_num]
        self.flux_unc = np.loadtxt(self.input_file + '_specerr.txt').T[:, ::-1][:, self.order_num]
        self.JD = np.loadtxt(self.input_file + '_JD.txt')
        itime = np.loadtxt(self.input_file + '_ExpLength.txt')
        self.JD += (itime / 2) / (86400)
        
        self.badpix = np.ones(len(self.flux), dtype=np.float64)
        
        # Normalize
        med_val = pcmath.weighted_median(self.flux, med_val=0.99)
        self.flux /= med_val
        self.flux_unc /= med_val
        
        
    @staticmethod
    def calculate_bc_info_all(forward_models, star_name, obs_name=None):
        """ Computes the bary-center information for all observations, specific to Mt. Kent.

        Args:
            forward_models (ForwardModels): The list of forward model objects.
            obs_name (str): The name of the observatory, not actually used.
            star_name (str): The name of the star to be queuried on SIMBAD.
        Returns:
            bjds (np.ndarray): The BJDs of the observations
            bc_vels (np.ndarray): The bary-center velocities of the observations.
        """
        # Import the barycorrpy module
        from barycorrpy import get_BC_vel
        from barycorrpy.utc_tdb import JDUTC_to_BJDTDB
        
        # Extract the jds
        jds = np.array([fwm.data.JD for fwm in forward_models], dtype=float)
        
        # BJDs
        bjds = JDUTC_to_BJDTDB(JDUTC=jds, starname=star_name.replace('_', ' '), lat=27.7977, longi=151.8554, alt=682)[0]
        
        # bc vels
        bc_vels = get_BC_vel(JDUTC=jds, starname=star_name.replace('_', ' '), lat=27.7977, longi=151.8554, alt=682)[0]
        
        for i in range(len(forward_models)):
            forward_models[i].data.set_bc_info(bjd=bjds[i], bc_vel=bc_vels[i])
        
        return bjds, bc_vels
    
    
class SpecDataMinervaNorth(SpecData1d):
    
    """ Class for extracted 1-dimensional spectra from the MINERVA North array.
    """
    
    def parse(self, forward_model):
        """Parses MINERVA North T1 data.
        """
        # Load the flux, flux unc, and bad pix arrays
        fits_data = fits.open(self.input_file)[0]
        fits_data.verify('fix')
        
        # The minerva telescope number to grab (1-4)
        self.tel_num = forward_model.tel_num
        
        # The flux
        self.flux = fits_data.data[self.tel_num - 1, self.order_num - 1, :].astype(np.float64)
        self.flux_unc = np.zeros_like(self.flux) + 1E-3
        
        # Normalize to 1.
        self.flux /= pcmath.weighted_median(self.flux, med_val=0.98)
        self.badpix = np.ones_like(self.flux)
        
        # JD from exposure meter. Sometimes it is not set in the header, so use the timing mid point in that case.        
        self.JD = float(fits_data.header['JD']) + float(fits_data.header['EXPTIME']) / (2 * 3600 * 24)
        
        
        
class SpecDataNIRSPEC(SpecData1d):
    """ Class for extracted 1-dimensional spectra from iSHELL on the NASA IRTF.
    """
        
    def parse(self, forward_model):
        """ Parses iSHELL data and computes the mid-exposure time (no exp meter for iSHELL).
        """
        
        # Load the flux, flux unc, and bad pix arrays
        fits_data = fits.open(self.input_file)[0]
        fits_data.verify('fix')
        oi = self.order_num - 1
        self.flux, self.flux_unc, self.badpix = fits_data.data[oi, 0, :, 0].astype(np.float64), fits_data.data[oi, 0, :, 1].astype(np.float64), fits_data.data[oi, 0, :, 2].astype(np.float64)

        # Flip the data so wavelength is increasing for iSHELL data
        #self.flux = self.flux[::-1]
        #self.badpix = self.badpix[::-1]
        #self.flux_unc = self.flux_unc[::-1]
        
        # Normalize to 99th percentile
        med_val = pcmath.weighted_median(self.flux, med_val=0.99)
        self.flux /= med_val
        self.flux_unc /= med_val
        
        # Define the JD of this observation using the mid point of the observation
        #self.JD = float(fits_data.header['TCS_UTC']) + 2400000.5 + float(fits_data.header['ITIME']) / (2 * 3600 * 24)
        
        # TEMP
        self.JD = 2455293.84426
        
        
class SpecDataPARVI_dev(SpecData1d):
    """ Class for extracted 1D spectra from PARVI using Chaz example txt file.
    """
        
    def parse(self):
        #txt file example uses one file per order.  this might not be what self.input_file is about.
        txt_data = np.genfromtxt(self.input_file).T
        self.default_wave_grid, self.flux, self.flux_unc = txt_data[3].astype(np.float64), txt_data[4].astype(np.float64), txt_data[5].astype(np.float64)
        #print("JWD sanity check default_wave_grid on load: ",self.default_wave_grid)

        # Create bad pix array, further cropped later
        self.badpix = np.ones(self.flux.size, dtype=np.float64)
        bad = np.where(~np.isfinite(self.flux) | (self.flux <= 0) | (self.flux_unc == 0) | (self.flux_unc > 0.5))[0]
        if bad.size > 0:
            self.badpix[bad] = 0

        #input data was normalized before application of simulated atmospheric effects.  renormalize flux and flux_unc to prevent problems in blaze fitting.
        factor = 1.0/np.mean(self.flux)
        self.flux*=factor
        self.flux_unc*=factor
        print("Renormalizing with factor of ",factor," for file ",self.input_file)
        
        # Convert wavelength grid to Angstroms, required!
        self.default_wave_grid *= 10
        # JWD I think chaz gave the wrong units.
        self.default_wave_grid *= 1000

        # use average observed time to get JD (all times should be equal)
        self.JD = np.mean(txt_data[0]).astype(np.float64)+2450000