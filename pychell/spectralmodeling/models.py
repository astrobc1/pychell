# Base Python
import copy
import os

# Maths
import numpy as np

# pychell
import pychell
import pychell.maths as pcmath
import pychell.spectralmodeling.rvcalc as pcrvcalc

# Plots
import matplotlib.pyplot as plt
try:
    plt.style.use(f"{os.path.dirname(pychell.__file__) + os.sep}gadfly_stylesheet.mplstyle")
except:
    print("Could not locate gadfly stylesheet, using default matplotlib stylesheet.")

# Optimize
from optimize.parameters import BoundedParameters, BoundedParameter

########################
#### SPECTRAL RANGE ####
########################

class SpectralRange:
    
    __slots__ = ['pixmin', 'pixmax', 'wavemin', 'wavemax', 'label', 'data_inds']
    
    def __init__(self, pixmin, pixmax, wavemin, wavemax, label=None):
        """Initiate a spectral region.

        Args:
            pixmin (int): The min pixel.
            pixmax (int): The max pixel.
            wavemin (float): The minimum wavelength
            wavemax (float): The maximum wavelength
            label (str, optional): A label for this region. Defaults to None.
        """
        self.pixmin = pixmin
        self.pixmax = pixmax
        self.wavemin = wavemin
        self.wavemax = wavemax
        self.label = label
        self.data_inds = np.arange(self.pixmin, self.pixmax + 1).astype(int)
        
    def __len__(self):
        return self.pix_len()
    
    def wave_len(self):
        return self.wavemax - self.wavemin
    
    def pix_len(self):
        return self.pixmax - self.pixmin + 1
        
    def pix_within(self, pixels, pad=0):
        good = np.where((pixels >= self.pixmin - pad) & (pixels <= self.pixmax + pad))[0]
        return good
        
    def wave_within(self, waves, pad=0):
        good = np.where((waves >= self.wavemin - pad) & (waves <= self.wavemax + pad))[0]
        return good
    
    def midwave(self):
        return self.wavemin + self.wave_len() / 2
    
    def midpix(self):
        return self.pixmin + self.pix_len() / 2
        
    def pix_per_wave(self):
        return (self.pixmax - self.pixmin) / (self.wavemax - self.wavemin)
    
    def __repr__(self):
        return f"Spectral Region: Pix: ({self.pixmin}, {self.pixmax}) Wave: ({self.wavemin}, {self.wavemax})"

##################################
#### COMPOSITE SPECTRAL MODEL ####
##################################

class IterativeSpectralForwardModel:
    """The primary container for an iterative spectral forward model problem.
    """
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, wls=None, continuum=None, lsf=None,
                 star=None,
                 tellurics=None,
                 gas_cell=None,
                 fringing=None,
                 order_num=None,
                 n_iterations=10,
                 oversample=8,
                 crop_pix=[200, 200]):
        """Initiate an iterative spectral forward model object.

        Args:
            wls (WavelengthSolution, optional): The wavelength solution model. Defaults to None.
            continuum (Continuum, optional): The continuum model. Defaults to None.
            lsf (LSF, optional): The LSF model. Defaults to None.
            star (AugmentedStar, optional): The Stellar model. Defaults to None.
            tellurics (Tellurics, optional): The telluric model. Defaults to None.
            gas_cell (GasCell, optional): The gas cell model. Defaults to None.
            fringing (FPCavityFringing, optional): The fringing model. Defaults to None.
            order_num (int): The order number.
            n_iterations (int, optional): The number of iterations, or number of times to augment the template(s). Defaults to 10.
            oversample (int, optional): The oversample factor of the model relative to the data, which is important for proper convolution. Defaults to 8.
            crop_pix (list, optional): How many pixels to crop on the left and right of the observation when ordered accordibg to wavelength. Defaults to [200, 200].
        """
        
        # The order number
        self.order_num = order_num
        
        # Model resolution
        self.oversample = oversample
        
        # Number of iterations
        self.n_iterations = n_iterations
        
        # Number of pixels to crop
        self.crop_pix = crop_pix
        
        # Model components
        self.wls = wls
        self.continuum = continuum
        self.lsf = lsf
        self.star = star
        self.tellurics = tellurics
        self.gas_cell = gas_cell
        self.fringing = fringing
        
    def init_templates(self, data):
        data0 = [d for d in data if d.is_good][0]
        data_wave_grid = data0.spec_module.estimate_wls(data0)
        good = np.where(data0.mask == 1)[0]
        if good.size == 0:
            raise ValueError(f"{data[0]} contains no good data")
        self.srange = SpectralRange(pixmin=good[0], pixmax=good[-1], wavemin=data_wave_grid[good[0]], wavemax=data_wave_grid[good[-1]])
        self.model_dl = (1 / self.srange.pix_per_wave()) / self.oversample
        self.model_wave = np.arange(self.srange.wavemin, self.srange.wavemax, self.model_dl)
        self.templates_dict = {}
        if self.star is not None:
            self.templates_dict["star"] = self.star.init_template(data, self.srange, self.model_dl)
        if self.gas_cell is not None:
            self.templates_dict["gas_cell"] = self.gas_cell.init_template(data, self.srange, self.model_dl)
        if self.tellurics is not None:
            self.templates_dict["tellurics"] = self.tellurics.init_template(data, self.srange, self.model_dl)
        
    def init_parameters(self, data):
        self.p0 = BoundedParameters()
        if self.wls is not None:
            self.p0.update(self.wls.init_parameters(data))
        if self.lsf is not None:
            self.p0.update(self.lsf.init_parameters(data))
        if self.continuum is not None:
            self.p0.update(self.continuum.init_parameters(data))
        if self.star is not None:
            self.p0.update(self.star.init_parameters(data))
        if self.gas_cell is not None:
            self.p0.update(self.gas_cell.init_parameters(data))
        if self.tellurics is not None:
            self.p0.update(self.tellurics.init_parameters(data))
        if self.fringing is not None:
            self.p0.update(self.fringing.init_parameters(data))
    
    ####################
    #### INITIALIZE ####
    ####################
    
    def initialize(self, p0, data, iter_index, stellar_templates):
        
        # Store the data
        self.data = data
        
        # Store the initial parameters
        self.p0 = p0
        
        # Init the models which may modify the templates
        if self.wls is not None:
            self.wls.initialize(self, iter_index)
        if self.continuum is not None:
            self.continuum.initialize(self, iter_index)
        if self.lsf is not None:
            self.lsf.initialize(self, iter_index)
        if self.star is not None:
            self.templates_dict['star'] = stellar_templates[iter_index]
            self.star.initialize(self, iter_index)
        if self.tellurics is not None:
            self.tellurics.initialize(self, iter_index)
        if self.gas_cell is not None:
            self.gas_cell.initialize(self, iter_index)
        if self.fringing is not None:
            self.fringing.initialize(self, iter_index)
    
    ##################
    #### BUILDERS ####
    ##################
        
    def build(self, pars, wave_final=None):
        
        # Alias model wave grid
        model_wave = self.model_wave
        
        # Alias models and templates dicts
        templates_dict = self.templates_dict
            
        # Init a model
        model_flux = np.ones_like(model_wave)

        # Star
        if self.star is not None:
            model_flux *= self.star.build(pars, templates_dict['star'], model_wave)
        
        # Gas Cell
        if self.gas_cell is not None:
            model_flux *= self.gas_cell.build(pars, templates_dict['gas_cell'], model_wave)
            
        # All tellurics
        if self.tellurics is not None and not self.tellurics.mask:
            model_flux *= self.tellurics.build(pars, templates_dict['tellurics'], model_wave)
        
        # Fringing from who knows what
        if self.fringing is not None:
            model_flux *= self.fringing.build(pars, model_wave)
            
        # Convolve
        if self.lsf is not None:

            model_flux = self.lsf.convolve_flux(model_flux, pars)
            
            # Renormalize model to remove degeneracy between blaze and lsf
            model_flux /= pcmath.weighted_median(model_flux, percentile=0.99)
            
        # Continuum
        if self.continuum is not None:
            model_flux *= self.continuum.build(pars, model_wave)

        # Generate the wavelength solution of the data
        if self.wls is not None:
            data_wave = self.wls.build(pars)

        # Interpolate high res model onto data grid
        if wave_final is None:
            model_flux = pcmath.cspline_interp(model_wave, model_flux, data_wave)
        else:
            model_flux = pcmath.cspline_interp(model_wave, model_flux, wave_final)
        
        # Return
        return data_wave, model_flux
    
    ###############
    #### MISC. ####
    ###############
    
    def summary(self, pars):
        s = ""
        if self.wls is not None:
            s += "--Wavelength Solution--:\n"
            for pname in self.wls.par_names:
                s += f"  {pars[pname]}\n"
        if self.continuum is not None:
            s += "--Continuum--:\n"
            for pname in self.continuum.par_names:
                s += f"  {pars[pname]}\n"
        if self.lsf is not None:
            s += "--LSF--:\n"
            for pname in self.lsf.par_names:
                s += f"  {pars[pname]}\n"
        if self.star is not None:
            s += "--Star--:\n"
            for pname in self.star.par_names:
                s += f"  {pars[pname]}\n"
        if self.tellurics is not None:
            s += "--Tellurics--:\n"
            for pname in self.tellurics.par_names:
                s += f"  {pars[pname]}\n"
        if self.gas_cell is not None:
            s += "--Gas Cell--:\n"
            for pname in self.gas_cell.par_names:
                s += f"  {pars[pname]}\n"
        return s

    def __repr__(self):
        s = f"Spectral Forward Model, {round(self.srange.wavemin, 1)} - {round(self.srange.wavemax, 1)} Å, oversample = {self.oversample}, \n"
        if self.wls is not None:
            s += f"  {self.wls}\n"
        if self.continuum is not None:
            s += f"  {self.continuum}\n"
        if self.lsf is not None:
            s += f"  {self.lsf}\n"
        if self.star is not None:
            s += f"  {self.star}\n"
        if self.tellurics is not None:
            s += f"  {self.tellurics}\n"
        if self.gas_cell is not None:
            s += f"  {self.gas_cell}\n"
        s = s[0:-1]
        return s

###################################
#### SPECTRAL MODEL COMPONENTS ####
###################################

from .components import *