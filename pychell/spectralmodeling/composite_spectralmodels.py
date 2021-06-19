# Base Python
import copy
import os

# Maths
import numpy as np

# Plots
import matplotlib.pyplot as plt
try:
    plt.style.use("gadfly_stylesheet")
except:
    print("Could not locate gadfly stylesheet, using default matplotlib stylesheet.")

# pychell
import pychell.maths as pcmath
import pychell.spectralmodeling.rvcalc as pcrvcalc
import pychell.spectralmodeling.spectralmodels as pcspecmodels
from pychell.spectralmodeling.spectralmodels import SpectralRegion

# Optimize
from optimize.models import Model
from optimize.knowledge import BoundedParameters, BoundedParameter

##################################
#### COMPOSITE SPECTRAL MODEL ####
##################################

class IterativeSpectralForwardModel(Model):
    """The primary container for a spectral forward model problem.
    """
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, data, blueprints, order_num, sregion, model_resolution, templates_path, wls_estimate):
        
        # Image order number
        self.order_num = order_num
        
        # Per component blueprints
        self.blueprints = blueprints
        
        # Spectral region
        self.sregion = sregion
        
        # Model resolution
        self.model_resolution = model_resolution
        self.model_dl = (1 / sregion.pix_per_wave()) / self.model_resolution
        
        # The templates path
        self.templates_path = templates_path
        
        # Initialize the models dict
        self._init_models(data, wls_estimate)
        
        # Initialize the templates dict
        self._init_templates(data)
        
        # Initialize the model parameters
        self._init_parameters(data)
        
        
    def _init_models(self, data, wls_estimate):
        
        # A dictionary to store model components
        self.models_dict = {}

        # First generate the wavelength solution model
        wave_class = getattr(pcspecmodels, self.blueprints["wavelength_solution"]["class"])
        
        # Must have a wavelength solution
        self.models_dict['wavelength_solution'] = wave_class(self.blueprints['wavelength_solution'], self.sregion, self.order_num, wls_estimate)
        
        # Define the LSF model if present
        if 'lsf' in self.blueprints:
            model_class_init = getattr(pcspecmodels, self.blueprints['lsf']['class'])
            self.models_dict['lsf'] = model_class_init(self.blueprints['lsf'], self.sregion, self.order_num, self.model_dl)
        
        # Generate the remaining model components from their blueprints and load any input templates
        # All remaining model components should subtype MultComponent or LSF
        for mkey in self.blueprints:
            
            if mkey in self.models_dict:
                continue
            
            blueprint = self.blueprints[mkey]
            
            # Construct the model
            model_class = getattr(pcspecmodels, blueprint['class'])
            self.models_dict[mkey] = model_class(blueprint, self.sregion, self.order_num)
            
        # The default high resolution grid for the wavelength solution
        self.model_wave = np.arange(self.sregion.wavemin - 5, self.sregion.wavemax + 5, self.model_dl)
        
    def _init_templates(self, data):
        self.templates_dict = {}
        for model_key in self.models_dict:
            if isinstance(self.models_dict[model_key], pcspecmodels.TemplateMult):
                self.templates_dict[model_key] = self.models_dict[model_key]._init_templates(data, self.templates_path, self.model_dl)
        
    def _init_parameters(self, data):
        self.p0 = BoundedParameters()
        for model_key in self.models_dict:
            self.p0.update(self.models_dict[model_key]._init_parameters(data))
        
    def _init_chunks(self, data):
        
        # The good pixels for the whole order for the first observation
        good = np.where(data[0].mask)[0]
        
        # The first and last good pixel
        order_pixmin, order_pixmax = good[0], good[-1]
        
        # The wavelength solution class
        wave_class = getattr(pcspecmodels, self.blueprints['wavelength_solution']['class'])
        
        # Store chunks in a list
        self.chunk_regions = []
        
        # The pixel stitch points
        stitch_points_pix = np.linspace(order_pixmin, order_pixmax, num=self.n_chunks + 1).astype(int)
        
        # An estimate to the wavelength solution
        wave_estimate = wave_class.estimate_order_wave(data[0], self.blueprints["wavelength_solution"])
        
        # The spectral region for the whole order
        self.sregion_order = SpectralRegion(order_pixmin, order_pixmax, wave_estimate[order_pixmin], wave_estimate[order_pixmax], label="order")
        
        # The model resolution grid spacing
        self.model_dl = (1 / self.sregion_order.pix_per_wave()) / self.model_resolution
        for ichunk in range(self.n_chunks):
            pixmin, pixmax = stitch_points_pix[ichunk], stitch_points_pix[ichunk + 1]
            wavemin, wavemax = wave_estimate[pixmin], wave_estimate[pixmax]
            self.chunk_regions.append(SpectralRegion(pixmin, pixmax, wavemin, wavemax, label=ichunk))
    
    ####################
    #### INITIALIZE ####
    ####################
    
    def initialize(self, p0, data, iter_index):
        
        # Store the data
        self.data = data
        
        # Store the initial parameters
        self.p0 = p0
        
        # Init the models which may modify the templates
        for model in self.models_dict.values():
            model.initialize(self.p0, self.data, self.templates_dict, iter_index)
            
        # Determine initial stellar vel if necessary
        if iter_index == 0 and "star" in self.models_dict and not self.models_dict["star"].from_flat:
            init_vel = pcrvcalc.brute_force_ccf_crude(self.p0, self.data, self)
            self.p0[self.models_dict["star"].par_names[0]].value = init_vel
    
    ##################
    #### BUILDERS ####
    ##################
        
    def build(self, pars, wave_final=None):
        
        # Alias model wave grid
        model_wave = self.model_wave
        
        # Alias models and templates dicts
        models_dict = self.models_dict
        templates_dict = self.templates_dict
            
        # Init a model
        model_flux = np.ones_like(model_wave)

        # Star
        if "star" in models_dict:
            model_flux *= models_dict['star'].build(pars, templates_dict['star'], model_wave)
        
        # Gas Cell
        if "gas_cell" in models_dict:
            model_flux *= models_dict['gas_cell'].build(pars, templates_dict['gas_cell'], model_wave)
            
        # All tellurics
        if "tellurics" in models_dict:
            model_flux *= models_dict['tellurics'].build(pars, templates_dict['tellurics'], model_wave)
        
        # Fringing from who knows what
        if "fringing" in models_dict:
            model_flux *= models_dict['fringing'].build(pars, model_wave)
            
        # Convolve
        if "lsf" in models_dict:
            model_flux = self.models_dict['lsf'].convolve_flux(model_flux, pars)
            
            # Renormalize model to remove degeneracy between blaze and lsf
            model_flux /= pcmath.weighted_median(model_flux, percentile=0.99)
            
        # Continuum
        if "continuum" in models_dict:
            model_flux *= self.models_dict['continuum'].build(pars, model_wave)

        # Generate the wavelength solution of the data
        if "wavelength_solution" in models_dict:
            data_wave = self.models_dict['wavelength_solution'].build(pars)

        # Interpolate high res model onto data grid
        if wave_final is None:
            model_flux_lr = pcmath.cspline_interp(model_wave, model_flux, data_wave)
        else:
            model_flux_lr = pcmath.cspline_interp(model_wave, model_flux, wave_final)
        
        # Return
        return data_wave, model_flux_lr
    
    ###############
    #### MISC. ####
    ###############
    
    def summary(self, pars):
        s = ""
        for mkey in self.models_dict:
            # Print the model string
            s += f"{self.models_dict[mkey].name}\n"
            # Sub loop over per model parameters
            for pname in self.models_dict[mkey].par_names:
                s += f"  {pars[pname]}\n"
        return s
    
    def __repr__(self):
        s = ""
        for mkey in self.models_dict:
            # Print the model string
            s += f"{self.models_dict[mkey].name}\n"
            # Sub loop over per model parameters
            for pname in self.models_dict[mkey].par_names:
                s += f"  {self.p0[pname]}\n"
        return s
    