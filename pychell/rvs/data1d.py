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
            
        # Sanity
        bad = np.where((self.flux < 0.05) | ~np.isfinite(self.flux) | (self.badpix == 0) | ~np.isfinite(self.badpix) | ~np.isfinite(self.flux_unc))[0]
        if bad.size > 0:
            self.flux[bad] = np.nan
            self.flux_unc[bad] = np.nan
            self.badpix[bad] = 0
            
        # Further flag any clearly deviant pixels
        flux_smooth = pcmath.median_filter1d(self.flux, width=7)
        bad = np.where(np.abs(flux_smooth - self.flux) > 0.3)[0]
        if bad.size > 0:
            self.flux[bad] = np.nan
            self.flux_unc[bad] = np.nan
            self.badpix[bad] = 0
        
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
        
        # BJD
        self.bjd = JDUTC_to_BJDTDB(JDUTC=self.JD, starname=star_name.replace('_', ' '), obsname=obs_name, ephemeris='de430', leap_update=True)[0]
        
        # bc vel
        self.bc_vel = get_BC_vel(JDUTC=jds, starname=star_name.replace('_', ' '), obsname=obs_name, ephemeris='de430', leap_update=True)[0]
    
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
      
    @staticmethod
    def star_from_simbad(star_name):
        star_dict = {}
        try:
            Simbad.add_votable_fields('PLX')
            Simbad.add_votable_fields('velocity')
            Simbad.add_votable_fields('pm')
            simbad_result = Simbad.query_object(star_name)
            ra_list = simbad_result['ra'].split(' ')
            star_dict['ra'] = (ra_list[0] + ra_list[1] / 60 + ra_list[2] / 3600) * 15
            dec_list = simbad_result['dec'].split(' ')
            star_dict['dec'] = dec_list[0] + dec_list[1] / 60 + dec_list[2] / 3600
            star_dict['px'] = simbad_result['PLX_VALUE'][0]
            star_dict['rv'] = simbad_result['RVZ_RADVEL'][0]
            star_dict['pmra'] = simbad_result['PMRA'][0]
            star_dict['pmdec'] = simbad_result['PMDEC'][0]
        except:
            raise ValueError("Could not parse simbad for " + star_name)
        
        return star_dict

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
        bjds, bc_vels = np.loadtxt(file, delimiter=',', unpack=True)
        return bjds, bc_vels
    
    
    @staticmethod
    def calculate_bc_info_all(forward_models, observatory, star_name):
        """ Computes the bary-center information for all observations.

        Args:
            forward_models (ForwardModels): The list of forward model objects.
            observatory (dict): A dictionary of observatory information. The name entry is looked up on EarthLocations.
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
        bjds = JDUTC_to_BJDTDB(JDUTC=jds, starname=star_name.replace('_', ' '), obsname=observatory['name'], ephemeris='de430', leap_update=True)[0]
        
        # bc vels
        bc_vels = get_BC_vel(JDUTC=jds, starname=star_name.replace('_', ' '), obsname=observatory['name'], ephemeris='de430', leap_update=True)[0]
        
        for i in range(forward_models.n_spec):
            forward_models[i].data.set_bc_info(bjd=bjds[i], bc_vel=bc_vels[i])
            
        return bjds, bc_vels

    
class iSHELL(SpecData1d):
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
        continuum = pcmath.weighted_median(self.flux, percentile=0.99)
        self.flux /= continuum
        self.flux_unc /= continuum
        
        # Define the JD of this observation using the mid point of the observation
        self.JD = float(fits_data.header['TCS_UTC']) + 2400000.5 + float(fits_data.header['ITIME']) / (2 * 3600 * 24)
        

class CHIRON(SpecData1d):
    """ Class for extracted 1-dimensional spectra from CHIRON on the SMARTS 1.5 m telescope.
    """
    def parse(self, forward_model):
        """ Parses CHIRON data and extracts the flux weighted midpoint of the exposure from the header if present, otherwise computes the mid exposure time. The flux uncertainty is not provided from CHIRON (?), so we assume all normalized uncertainties to arbitrarily be 0.001 (uniform). The wavelength grid provided by the ThAr lamp is provided in the wave_grid attribute.
        """
        
        # Load the flux, flux unc, and bad pix arrays
        fits_data = fits.open(self.input_file)[0]
        fits_data.verify('fix')
        
        self.default_wave_grid, self.flux = fits_data.data[self.order_num - 1, :, 0].astype(np.float64), fits_data.data[self.order_num - 1, :, 1].astype(np.float64)
        self.flux /= pcmath.weighted_median(self.flux, percentile=0.98)
        
        # For CHIRON, generate a dumby uncertainty grid and a bad pix array that will be updated or used
        self.flux_unc = np.zeros_like(self.flux) + 1E-3
        self.badpix = np.ones_like(self.flux)
        
        # JD from exposure meter. Sometimes it is not set in the header, so use the timing mid point in that case.
        if not (fits_data.header['EMMNWB'][0:2] == '00'):
            self.JD = Time(fits_data.header['EMMNWB'].replace('T', ' '), scale='utc').jd
        else:
            self.JD = Time(fits_data.header['DATE-OBS'].replace('T', ' '), scale='utc').jd + float(fits_data.header['EXPTIME']) / (2 * 3600 * 24)
        
        
class PARVI(SpecData1d):
    
    """ Class for extracted 1-dimensional spectra from PARVI.
    """
    def parse(self, forward_model):
        
        # Load the flux, flux unc, and bad pix arrays. Also load the known wavelength grid for a starting point
        fits_file = fits.open(self.input_file)
        data = fits_file[1].data
        header = fits_file[0].header
        oi = self.order_num - 1
        
        # 0,1,2 are wave,counts,counts_var
        self.default_wave_grid, self.flux, self.flux_unc = data[0, oi, :].astype(np.float64), data[7, oi, :].astype(np.float64), data[8, oi, :].astype(np.float64)
        
        # Normalize according to 98th percentile in flux
        continuum = pcmath.weighted_median(self.flux, percentile=0.98)
        self.flux /= continuum
        self.flux_unc /= continuum
        
        # Create bad pix array, further cropped later according to crop_pix
        self.badpix = np.ones(self.flux.size, dtype=np.float64)
        bad = np.where(~np.isfinite(self.flux))[0]
        if bad.size > 0:
            self.badpix[bad] = 0
        
        # Convert wavelength grid to Angstroms, required!
        self.default_wave_grid *= 10
        
        # JD (mid expsosure) from header
        self.JD = float(header['MJD']) # + 2450000
        
        
class MinervaAustralis(SpecData1d):
        
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
        continuum = pcmath.weighted_median(self.flux, percentile=0.99)
        self.flux /= continuum
        self.flux_unc /= continuum
        
        
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
    
    
class MinervaNorth(SpecData1d):
    
    """ Class for extracted 1-dimensional spectra from the MINERVA North array.
    """
    
    def parse(self, forward_model):
        """Parses MINERVA North data.
        """
        # Load the flux, flux unc, and bad pix arrays
        fits_data = fits.open(self.input_file)[0]
        fits_data.verify('fix')
        
        self.tel_num = int(self.input_file[-6])
        
        # The Thar wave grid, flux, flux unc, and mask
        self.default_wave_grid, self.flux, self.flux_unc, self.badpix = fits_data.data[self.order_num - 1, :, 0].astype(np.float64), fits_data.data[self.order_num - 1, :, 1].astype(np.float64), fits_data.data[self.order_num - 1, :, 2].astype(np.float64), fits_data.data[self.order_num - 1, :, 3].astype(np.float64)

        self.flux_unc = np.zeros(self.flux.size) + 1E-3
        
        # Normalize to 1.
        continuum = pcmath.weighted_median(self.flux, percentile=0.98)
        self.flux /= continuum
        self.flux_unc /= continuum
        
        # JD from exposure meter. Sometimes it is not set in the header, so use the timing mid point in that case.        
        try:
            self.JD = float(fits_data.header['FLUXMID' + str(self.tel_num)])
        except:
            print('Warning: Using non weighted exposure mid point for ' + self.base_input_file)
            self.JD = float(fits_data.header['JD']) + float(fits_data.header['EXPTIME']) / (2 * 3600 * 24)
        
        
        
class NIRSPEC(SpecData1d):
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
        continuum = pcmath.weighted_median(self.flux, percentile=0.99)
        self.flux /= continuum
        self.flux_unc /= continuum
        
        # Define the JD of this observation using the mid point of the observation
        #self.JD = float(fits_data.header['TCS_UTC']) + 2400000.5 + float(fits_data.header['ITIME']) / (2 * 3600 * 24)
        
        # TEMP
        self.JD = 2455293.84426
        
class Simulated(SpecData1d):
    
    """ Simulated Data for internal testing.
    """
    def parse(self, forward_model):
        
        # Load the flux, flux unc, and bad pix arrays. Also load the known wavelength grid for a starting point
        fits_data = fits.open(self.input_file)[0]
        #oi = self.order_num - 1
        self.default_wave_grid = fits_data.data[:, 0].astype(np.float64)
        self.flux = fits_data.data[:, 1].astype(np.float64)
        self.flux_unc = np.zeros_like(self.flux) + 1E-3
        
        # Normalize according to 98th percentile in flux
        #continuum = pcmath.weighted_median(self.flux, percentile=0.99)
        #self.flux /= continuum
        
        # Create bad pix array, further cropped later
        self.badpix = np.ones(self.flux.size, dtype=np.float64)
        bad = np.where(~np.isfinite(self.flux) | (self.flux == 0) | (self.flux_unc == 0) | (self.flux_unc > 0.5))[0]
        if bad.size > 0:
            self.badpix[bad] = 0

        self.JD = float(fits_data.header['JD'])
        
        
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
        #bjds = JDUTC_to_BJDTDB(JDUTC=jds, ra=269.4520820833333, dec=4.693364166666667, pmra=-802.803, pmdec=10362.542, px=547.4506, rv=-110510.0, epoch=2451545.0, obsname=obs_name)[0]
        
        #bjds = JDUTC_to_BJDTDB(JDUTC=jds, obsname=obs_name, starname = star_name)[0]
        
        # bc vels
        #bc_vels = get_BC_vel(JDUTC=jds, ra=269.4520820833333, dec=4.693364166666667, pmra=-802.803, pmdec=10362.542, px=547.4506, rv=-110510.0, epoch=2451545.0, obsname=obs_name)[0]
        #bc_vels = get_BC_vel(JDUTC=jds, obsname=obs_name, starname = star_name)[0]
        bjds = JDUTC_to_BJDTDB(JDUTC=jds, ra=269.4520820833333, dec=4.693364166666667, pmra=-802.803, pmdec=10362.542, px=547.4506, rv=-110510.0, epoch=2451545.0, obsname=obs_name, ephemeris='de430', leap_update=True)[0]
        bc_vels = get_BC_vel(JDUTC=jds, ra=269.4520820833333, dec=4.693364166666667, pmra=-802.803, pmdec=10362.542, px=547.4506, rv=-110510.0, epoch=2451545.0, obsname=obs_name, ephemeris='de430', leap_update=True)[0]
        
        
        for i in range(forward_models.n_spec):
            forward_models[i].data.set_bc_info(bjd=bjds[i], bc_vel=bc_vels[i])
            
        return bjds, bc_vels