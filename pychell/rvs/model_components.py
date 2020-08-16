# Python built in modules
from collections import OrderedDict
from abc import ABC, abstractmethod  # Abstract classes
from pdb import set_trace as stop  # debugging
import inspect
import glob

# Science/math
from scipy import constants as cs  # cs.c = speed of light in m/s
import numpy as np  # Math, Arrays
import sys
from scipy.special import legendre
import scipy.interpolate  # Cubic, Akima interpolation

# Graphics
import matplotlib.pyplot as plt

# llvm
from numba import njit, jit, jitclass
import numba

# User defined
import pychell.maths as pcmath
import pychell.rvs.template_augmenter as pcaugmenter
import optimparameters.parameters as OptimParameters


class SpectralComponent:
    """Base class for a general spectral component model.

    Attributes:
        blueprint (dict): The blueprints to construct this component from.
        order_num (int): The image order number.
        enabled (bool): Whether or not this model is enabled.
        n_delay (int): The number of iterations to delay this model component.
        name (str): The name of this model.
        base_par_names (str): The base parameter names of the parameters for this model.
        par_names (str): The full parameter names are name + _ + base_par_names.
    """

    def __init__(self, forward_model, blueprint):
        """ Base initialization for a model component.

        Args:
            blueprint (dict): The blueprints to construct this component from.
            order_num (int): The image order number.
        """
        # Base class for spectral model components
        # Store the blueprint so it doesn't need to be passed around
        self.blueprint = blueprint

        # Default enabled, user can further choose to disable after calling super()
        if 'n_delay' in blueprint:
            self.n_delay = blueprint['n_delay']
        else:
            self.n_delay = 0
        
        # Whether or not to enable this model at the start    
        self.enabled = not (self.n_delay > 0)

        # The order number for this model
        self.order_num = forward_model.order_num

        # The blueprint must contain a name for this model
        self.name = blueprint['name']
        
        # Parameter names
        self.base_par_names = []
        self.par_names = []
        
        # The wavelength bounds for this model in the lab frame
        self.wave_bounds = forward_model.wave_bounds

    # Must implement a build method
    def build(self, pars, *args, **kwargs):
        raise NotImplementedError("Must implement a build method for this Spectral Component")

    # Must implement a build_fake method if ever disabling model
    def build_fake(self, *args, **kwargs):
        raise NotImplementedError("Must implement a build fake method for this Spectral Component")

    # Called after each iteration, may overload.
    def update(self, forward_model, iter_index):
        """Updates this model component given the iteration index and the n_delay attribute.

        Args:
            forward_model (ForwardModel): The forward model this model belongs to.
            iter_index (int): The iteration index.
        """
        index_offset = 0 if forward_model.models_dict['star'].from_synthetic else 1
        if iter_index + index_offset == self.n_delay and not self.enabled:
            self.enabled = True
            for pname in self.par_names:
                forward_model.initial_parameters[pname].vary = True

    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'
    
    def init_parameters(self, forward_model):
        pass
        
        
    def estimate_parameters(self, forward_model):
        raise NotImplementedError("Must implement an estimate_parameters method for this class")
        
    def estimate(self, forward_model):
        return self.build(forward_model.initial_parameters)
    
    def init_optimize(self, forward_model):
        pass
    


class MultModelComponent(SpectralComponent):
    """ Base class for a multiplicative (or log-additive) spectral component.

    Attributes:
        wave_bounds (list): The approximate left and right wavelength endpoints of the considered data.
        base_par_names (list): The base parameter names (constant) for this model.
        par_names (list): The full parameter names for this specific run.
    """

    def __init__(self, forward_model, blueprint):
        """Default constructor for a multiplicative model component.

        Args:
            blueprint (dict): The dictionary needed to construct this model component.
            wave_bounds (list): A list of the approximate min and max wavelength bounds.
            order_num (int, optional): The order number. Defaults to None.
        """
        # Call super method
        super().__init__(forward_model, blueprint)

    # Effectively no model
    def build_fake(self, nx):
        """Returns an array of ones.

        Args:
            nx (int): The number of points to build a model with.

        Returns:
            np.ndarray: An array of ones.
        """
        return np.ones(nx)


class EmpiricalMult(MultModelComponent):
    """ Base class for an empirically derived multiplicative (or log-additive) spectral component (i.e., based purely on parameters, no templates involved). As of now, this is purely a node in the Type tree and provides no additional functionality.
    """

    def __init__(self, forward_model, blueprint):
        """Default constructor for an empirical multiplicative model component.

        Args:
            blueprint (dict): The dictionary needed to construct this model component.
            wave_bounds (list): A list of the approximate min and max wavelength bounds.
            order_num (int, optional): The order number. Defaults to None.
        """
        # Call super method
        super().__init__(forward_model, blueprint)


class TemplateMult(MultModelComponent):
    """ A base class for a template based multiplicative model.

    Attributes:
        input_file (str): If provided, stores the full path + filename of the input file.
    """

    def __init__(self, forward_model, blueprint):
        """Constructs a multiplicative model that uses a template.

        Args:
            blueprint (dict): The dictionary needed to construct this model component.
            wave_bounds (list): A list of the approximate min and max wavelength bounds.
            order_num (int, optional): The order number. Defaults to None.
        """
        # Call super method
        super().__init__(forward_model, blueprint)

        # By default, set input_file. Some models (like tellurics) ignore this
        if 'input_file' in blueprint and blueprint['input_file'] is not None:
            self.input_file = forward_model.templates_path + blueprint['input_file']
        else:
            self.input_file = None


#### Blaze Models ####

class PolyBlaze(EmpiricalMult):
    """  Blaze transmission model through a polynomial and/or splines, ideally used after a flat field correction or after remove_continuum but not required.
    
    .. math:
        B(\\lambda) = (\\sum_{k=0}^{N} a_{i} \\lambda^{k} ) sinc(b (\\lambda - \\lambda_{B}))^{2d}
    

    Attributes:
        poly_order (int): The polynomial order.
        n_splines (int): The number of wavelength splines.
        blaze_wave_estimate (bool): The estimate of the blaze wavelegnth. If not provided, defaults to the average of the wavelength grid provided in the build method.
        spline_set_points (np.ndarray): The location of the spline knots.
    """

    def __init__(self, forward_model, blueprint):
        
        # Super
        super().__init__(forward_model, blueprint)
        
        # The polynomial order
        self.poly_order = blueprint['poly_order']
        self.n_poly_pars = self.poly_order + 1

        # The estimate of the blaze wavelength
        if 'blaze_wavelengths' in blueprint:
            self.blaze_wave_estimate = blueprint['blaze_wavelengths'][self.order_num - 1]
        else:
            self.blaze_wave_estimate = None
            
        # Parameter names
        self.base_par_names = []
        for i in range(self.n_poly_pars):
            self.base_par_names.append('_poly_' + str(i)) # starts at zero
                
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, wave_final):
        
        # If not enabled, return ones
        if not self.enabled:
            return self.build_fake(wave_final.size)
        
        # Blaze wavelength or average
        blaze_wavelength = self.blaze_wave_estimate if self.blaze_wave_estimate is not None else np.nanmean(wave_final)
        
        # The polynomial coeffs
        poly_pars = np.array([pars[self.par_names[i]].value for i in range(self.poly_order + 1)])
        
        # Build polynomial
        poly_blaze = np.polyval(poly_pars[::-1], wave_final - blaze_wavelength)
        
        return poly_blaze

    def init_parameters(self, forward_model):
        
        # Poly parameters
        for i in range(self.n_poly_pars):
            pname = 'poly_' + str(i)
            forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint[pname][1], minv=self.blueprint[pname][0], maxv=self.blueprint[pname][2], vary=self.enabled))

    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'

    
    
class SplineBlaze(EmpiricalMult):
    """  Blaze transmission model through a polynomial and/or splines, ideally used after a flat field correction or after remove_continuum but not required.
    
    .. math:
        B(\\lambda) = (\\sum_{k=0}^{N} a_{i} \\lambda^{k} ) sinc(b (\\lambda - \\lambda_{B}))^{2d}
    

    Attributes:
        poly_order (int): The polynomial order.
        n_splines (int): The number of wavelength splines.
        blaze_wave_estimate (bool): The estimate of the blaze wavelegnth. If not provided, defaults to the average of the wavelength grid provided in the build method.
        spline_set_points (np.ndarray): The location of the spline knots.
    """

    def __init__(self, forward_model, blueprint):
        
        # Super
        super().__init__(forward_model, blueprint)

        # The number of spline knots is n_splines + 1
        self.n_splines = blueprint['n_splines']
        if self.n_splines == 0:
            self.enabled = False
            self.n_spline_pars = 0
        else:
            self.n_spline_pars = self.n_splines + 1

        # Set the spline parameter names and knots
        self.spline_wave_set_points = np.linspace(self.wave_bounds[0], self.wave_bounds[1], num=self.n_splines + 1)
        for i in range(self.n_splines+1):
            self.base_par_names.append('_spline_' + str(i+1))
                
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, wave_final):
        
        # If not enabled, return ones
        if not self.enabled:
            return self.build_fake(wave_final.size)

        # Get the spline parameters
        spline_pars = np.array([pars[self.par_names[i]].value for i in range(self.n_spline_pars)], dtype=np.float64)

        # Build
        spline_blaze = scipy.interpolate.CubicSpline(self.spline_wave_set_points, spline_pars, extrapolate=False, bc_type='not-a-knot')(wave_final)
        
        return spline_blaze

    def init_parameters(self, forward_model):
        
        # Estimate the continuum
        wave = forward_model.models_dict['wavelength_solution'].build(forward_model.initial_parameters)
        log_continuum = pcaugmenter.fit_continuum_wobble(wave, np.log(forward_model.data.flux), forward_model.data.badpix, order=4, nsigma=[0.25, 3.0], maxniter=50)

        continuum = np.exp(log_continuum)
        good = np.where(np.isfinite(continuum))[0]

        # Spline parameters
        for i in range(self.n_spline_pars):
            if forward_model.remove_continuum:
                forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint['spline'][1] + 1, minv=self.blueprint['spline'][0] + 1, maxv=self.blueprint['spline'][2] + 1, vary=self.enabled))
            else:
                k = np.argmin(np.abs(wave[good] - self.spline_wave_set_points[i])) + good[0]
                forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=continuum[k], minv=continuum[k] + self.blueprint['spline'][0], maxv=continuum[k] + self.blueprint['spline'][2], vary=self.enabled))
                
    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'


class SincBlaze(EmpiricalMult):
    """ A full blaze transmission model.
    
    .. math:
        B(\\lambda) = (\\sum_{k=0}^{N} a_{i} \\lambda^{k} ) sinc(b (\\lambda - \\lambda_{B} + c))^{2d}

    Attributes:
        poly_order (int): The polynomial order.
        blaze_wave_estimate (bool): The estimate of the blaze wavelegnth. If not provided, defaults to the average of the wavelength grid provided in the build method. For this model, this is likely insufficient and robust estimates should be provided.
    """

    def __init__(self, forward_model, blueprint):

        # Super
        super().__init__(forward_model, blueprint)

        # The base parameter names
        self.base_par_names = ['_b', '_c', '_d']

        # The estimate of the blaze wavelength
        if 'blaze_wavelengths' in blueprint:
            self.blaze_wave_estimate = blueprint['blaze_wavelengths'][self.order_num - 1]
        else:
            self.blaze_wave_estimate = None
            
        # Polynomial
        self.poly_order = blueprint['poly_order']
        self.n_poly_pars = self.poly_order + 1

        # Set the polynomial parameter names
        if self.n_poly_pars > 0:
            for i in range(self.n_poly_pars):
                self.base_par_names.append('_a_' + str(i))

        self.par_names = [self.name + s for s in self.base_par_names]
    
    def build(self, pars, wave_final):
        
        # Sinc pars
        b = pars[self.par_names[0]].value
        c = pars[self.par_names[1]].value
        d = pars[self.par_names[2]].value
        
        # Polynomial pars
        poly_pars = np.array([pars[self.par_names[i + 3]] for i in range(self.n_poly_pars)])
        
        # Blaze wavelength or average
        blaze_wavelength = self.blaze_wave_estimate if self.blaze_wave_estimate is not None else np.nanmean(wave_final)
        
        # Construct sinc model
        sinc_blaze = np.abs(np.sinc(b * (wave_final - blaze_wavelength + c)))**(2 * d)
        
        # Construct polynomial
        poly_blaze = np.polyval(wave_final - blaze_wavelength)
        
        # Return product
        return blaze_base * blaze_spline

    def init_parameters(self, forward_model):
        
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['base_b'][1], minv=self.blueprint['base_b'][0], maxv=self.blueprint['base_b'][2], mcmcscale=0.1, vary=True))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[2], value=self.blueprint['base_c'][1], minv=self.blueprint['base_c'][0], maxv=self.blueprint['base_c'][2], mcmcscale=0.1, vary=True))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[3], value=self.blueprint['base_d'][1], minv=self.blueprint['base_d'][0], maxv=self.blueprint['base_d'][2], mcmcscale=0.1, vary=True))
        if self.n_splines > 0:
            for i in range(self.n_splines + 1):
                forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(
                    name=self.par_names[i+4], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], mcmcscale=0.001, vary=self.splines_enabled))


#### Gas Cell ####

class GasCell(TemplateMult):
    """ A gas cell model which is consistent across orders.
    """

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        self.base_par_names = ['_shift', '_depth']

        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, template, wave_final):
        wave, flux = template[:, 0], template[:, 1]
        if self.enabled:
            wave = wave + pars[self.par_names[0]].value
            flux = flux ** pars[self.par_names[1]].value
            return np.interp(wave_final, wave, flux, left=flux[0], right=flux[-1])
        
        else:
            return self.build_fake(wave_final.size)

    def load_template(self, forward_model):
        print('Loading in Gas Cell Template', flush=True)
        pad = 1
        template = np.load(self.input_file)
        wave, flux = template['wave'], template['flux']
        good = np.where((wave > self.wave_bounds[0] - pad) & (wave < self.wave_bounds[1] + pad))[0]
        wave, flux = wave[good], flux[good]
        flux /= pcmath.weighted_median(flux, percentile=0.999)
        template = np.array([wave, flux]).T
        return template

    def init_parameters(self, forward_model):
        
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['shift'][1], minv=self.blueprint['shift'][0], maxv=self.blueprint['shift'][2], mcmcscale=0.1, vary=True))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['depth'][1], minv=self.blueprint['depth'][0], maxv=self.blueprint['depth'][2], mcmcscale=0.001, vary=True))
        
    def init_optimize(self, forward_model):
        wave, flux = forward_model.templates_dict['gas_cell'][:, 0], forward_model.templates_dict['gas_cell'][:, 1]
        flux_interp = scipy.interpolate.CubicSpline(wave, flux, extrapolate=False)(forward_model.templates_dict['star'][:, 0])
        flux_conv = forward_model.models_dict['lsf'].convolve_flux(flux_interp, pars=forward_model.initial_parameters)
        forward_model.templates_dict['gas_cell'][:, 1] /= pcmath.weighted_median(flux_conv, percentile=0.99)

class GasCellCHIRON(TemplateMult):
    """ A gas cell model for CHIRON.
    """

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        self.base_par_names = ['_shift', '_depth']

        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, template, wave_final):
        wave, flux = template[:, 0], template[:, 1]
        if self.enabled:
            wave = wave + pars[self.par_names[0]].value
            flux = flux ** pars[self.par_names[1]].value
            return np.interp(wave_final, wave, flux, left=flux[0], right=flux[-1])
        else:
            return self.build_fake(wave_final.size)

    def load_template(self, forward_model):
        print('Loading in Gas Cell Template ...', flush=True)
        pad = 1
        template = np.load(self.input_file)
        wave, flux = template['wave'], template['flux']
        good = np.where((wave > self.wave_bounds[0] - pad) & (wave < self.wave_bounds[1] + pad))[0]
        wave, flux = wave[good], flux[good]
        flux /= pcmath.weighted_median(flux, percentile=0.999)
        template = np.array([wave, flux]).T
        return template

    def init_parameters(self, forward_model):
        
        shift = self.blueprint['shifts'][self.order_num - 1]
        depth = self.blueprint['depth']
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=shift, minv=shift - self.blueprint['shift_range'][0], maxv=shift + self.blueprint['shift_range'][1], mcmcscale=0.1, vary=True))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=depth[1], minv=depth[0], maxv=depth[2], mcmcscale=0.001, vary=self.enabled))
        
    def init_optimize(self, forward_model):
        wave, flux = forward_model.templates_dict['gas_cell'][:, 0], forward_model.templates_dict['gas_cell'][:, 1]
        flux_interp = scipy.interpolate.CubicSpline(wave, flux, extrapolate=False)(forward_model.templates_dict['star'][:, 0])
        flux_conv = forward_model.models_dict['lsf'].convolve_flux(flux_interp, pars=forward_model.initial_parameters)
        forward_model.templates_dict['gas_cell'][:, 1] /= pcmath.weighted_median(flux_conv, percentile=0.99)


#### Star ####

class Star(TemplateMult):
    """ A star model which may or may not have started from a synthetic template.
    
    Attr:
        from_synthetic (bool): Whether or not this model started from a synthetic template or not.
    """

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        self.base_par_names = ['_vel']

        if hasattr(self, 'input_file') and self.input_file is not None:
            self.from_synthetic = True
            self.n_delay = 0
        else:
            self.from_synthetic = False
            self.enabled = False
            self.n_delay = 1
        
        # Update parameter names
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, template, wave_final):
        wave, flux = template[:, 0], template[:, 1]
        if self.enabled:
            wave_shifted = wave * np.exp(pars[self.par_names[0]].value / cs.c)
            return np.interp(wave_final, wave_shifted, flux, left=np.nan, right=np.nan)
        else:
            return self.build_fake(wave_final.size)

    def update(self, forward_model, iter_index):
        super().update(forward_model, iter_index)

    def init_parameters(self, forward_model):
        
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=-1*forward_model.data.bc_vel, minv=self.blueprint['vel'][0], maxv=self.blueprint['vel'][2], mcmcscale=0.1, vary=self.enabled))

    def load_template(self, forward_model):
        pad = 15
        wave_even = np.linspace(self.wave_bounds[0] - pad, self.wave_bounds[1] + pad, num=forward_model.n_model_pix)
        if self.from_synthetic:
            print('Loading in Synthetic Stellar Template', flush=True)
            template_raw = np.loadtxt(self.input_file, delimiter=',')
            wave, flux = template_raw[:, 0], template_raw[:, 1]
            flux_interp = scipy.interpolate.CubicSpline(wave, flux, extrapolate=False, bc_type='not-a-knot')(wave_even)
            flux_interp /= pcmath.weighted_median(flux_interp, percentile=0.999)
            template = np.array([wave_even, flux_interp]).T
        else:
            template = np.array([wave_even, np.ones(wave_even.size)]).T

        return template
    
    def init_optimize(self, forward_model):
        wave, flux = forward_model.templates_dict['star'][:, 0], forward_model.templates_dict['star'][:, 1]
        flux_conv = forward_model.models_dict['lsf'].convolve_flux(flux, pars=forward_model.initial_parameters)
        forward_model.templates_dict['star'][:, 1] /= pcmath.weighted_median(flux_conv, percentile=0.99)


#### Tellurics ####

class TelluricsTAPAS(TemplateMult):
    """ A telluric model based on Templates obtained from TAPAS. These templates should be pre-fetched from TAPAS and specific to the observatory. Each species has a unique depth, but the model is locked to a common Doppler shift.

    Attributes:
        species (list): The names (strings) of the telluric species.
        n_species (int): The number of telluric species.
        species_enabled (dict): A dictionary with species as keys, and boolean values for items (True=enabled, False=disabled)
        species_input_files (list): A list of input files (strings) for the individual species.
    """

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        self.base_par_names = ['_vel']

        if len(blueprint['species'].keys()) == 0:
            self.species = []
            self.enabled = False
        else:
            self.species = list(blueprint['species'].keys())
            self.n_species = len(self.species)
            self.species_enabled = {}
            self.species_input_files = {}
            for itell in range(self.n_species):
                self.species_input_files[self.species[itell]] = forward_model.templates_path + blueprint['species'][self.species[itell]]['input_file']
                self.species_enabled[self.species[itell]] = True
                self.base_par_names.append('_' + self.species[itell] + '_depth')
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, templates, wave_final):
        if self.enabled:
            flux = np.ones(wave_final.size, dtype=np.float64)
            for i in range(self.n_species):
                if self.species_enabled[self.species[i]]:
                    flux *= self.build_single_species(pars, templates, self.species[i], i, wave_final)
            return flux
        else:
            return self.build_fake(wave_final.size)

    def build_single_species(self, pars, templates, single_species, species_i, wave_final):
        shift = pars[self.par_names[0]].value
        depth = pars[self.par_names[species_i + 1]].value
        wave, flux = templates[single_species][:, 0], templates[single_species][:, 1]
        flux = flux ** depth
        wave_shifted = wave * np.exp(shift / cs.c)
        return np.interp(wave_final, wave_shifted, flux, left=flux[0], right=flux[-1])

    def init_parameters(self, forward_model):
        
        # Components
        for itell, tell in enumerate(self.species):
            max_range = np.nanmax(forward_model.templates_dict['tellurics'][tell][:, 1]) - np.nanmin(forward_model.templates_dict['tellurics'][tell][:, 1])
            if max_range > 0.015:
                self.species_enabled[tell] = True
            else:
                self.species_enabled[tell] = False
            forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[itell+1], value=self.blueprint['species'][self.species[itell]]['depth'][1], minv=self.blueprint['species'][self.species[itell]]['depth'][0], maxv=self.blueprint['species'][self.species[itell]]['depth'][2], mcmcscale=0.1, vary=self.species_enabled[tell]))
            
        # Shift
        if np.any([self.species_enabled[tell] for tell in self.species_enabled]):
            v = True
        else:
            v = False
            self.enabled = False
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['vel'][1], minv=self.blueprint['vel'][0], maxv=self.blueprint['vel'][2], mcmcscale=0.1, vary=v))
        
        if not self.enabled:
            self.n_delay = int(1E3)

    def update(self, forward_model, iter_index):
        super().update(forward_model, iter_index)

    def load_template(self, forward_model):
        templates = {}
        pad = 1
        for i in range(self.n_species):
            print('Loading in Telluric Template For ' + self.species[i], flush=True)
            template = np.load(self.species_input_files[self.species[i]])
            wave, flux = template['wave'], template['flux']
            good = np.where((wave > self.wave_bounds[0] - pad) & (wave < self.wave_bounds[1] + pad))[0]
            wave, flux = wave[good], flux[good]
            flux /= pcmath.weighted_median(flux, percentile=0.999)
            templates[self.species[i]] = np.array([wave, flux]).T
        return templates

    def __repr__(self):
        ss = ' Model Name: ' + self.name + ', Species: ['
        for tell in self.species_enabled:
            if self.species_enabled[tell]:
                ss += tell + ': Active, '
            else:
                ss += tell + ': Deactive, '
        ss = ss[0:-2]
        return ss + ']'
    
    def init_optimize(self, forward_model):
        ts = forward_model.templates_dict['tellurics']
        for t in ts:
            wave, flux = ts[t][:, 0], ts[t][:, 1]
            flux_interp = scipy.interpolate.CubicSpline(wave, flux, extrapolate=False)(forward_model.templates_dict['star'][:, 0])
            flux_conv = forward_model.models_dict['lsf'].convolve_flux(flux_interp, pars=forward_model.initial_parameters)
            ts[t][:, 1] /= pcmath.weighted_median(flux_conv, percentile=0.999)


#### LSF ####

class LSF(SpectralComponent):
    """ A base class for an LSF (line spread function) model.

    Attributes:
        compress (int): The number of lsf points is equal to the number of model pix / compress.
        dl (float): The step size of the high resolution fidicual wavelength grid the model is convolved on. Must be evenly spaced.
        nx_model (float): The number of model pixels in the high resolution fidicual wavelength grid.
        nx (int): The number of points in the lsf grid.
        x (np.ndarray): The lsf grid.
        default_lsf (np.ndarray): The default LSF to use or start from. Defaults to None.
    """

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        # The resolution of the model grid (dl = grid spacing)
        self.dl = forward_model.dl
        
        if 'compress' in blueprint:
            self.compress = blueprint['compress']
        else:
            self.compress = 64

        # The number of model pixels
        self.nx_model = forward_model.n_model_pix
        
        # The number of points in the grid
        self.nx_lsf = int(self.nx_model / self.compress)

        # The actual LSF x grid, force to be odd
        if self.nx_lsf % 2 != 1:
            self.nx_lsf += 1
        
        # Padding both left and right
        self.n_pad_model = int(np.floor(self.nx_lsf / 2))
        
        # Grid for LSF
        self.x = np.arange(-np.floor(self.nx_lsf / 2), np.floor(self.nx_lsf / 2) + 1, 1) * self.dl
        
        # Set the default LSF if provided
        if hasattr(forward_model.data, 'default_lsf') and forward_model.data.default_lsf is not None:
            self.default_lsf_raw = forward_model.data.default_lsf
            self.default_lsf = scipy.interpolate.CubicSpline(self.default_lsf_raw[:, 0], self.default_lsf_raw[:, 1])(self.x)
        else:
            self.default_lsf = None

    # Returns a delta function
    def build_fake(self):
        delta = np.zeros(self.nx_lsf, dtype=float)
        delta[int(np.floor(self.nx_lsf / 2))] = 1.0
        return delta

    # Convolves the flux
    def convolve_flux(self, raw_flux, pars=None, lsf=None):
        if lsf is None and pars is None:
            raise ValueError("Cannot construct LSF with no parameters")
        if not self.enabled:
            return raw_flux
        if lsf is None:
            lsf = self.build(pars)
        padded_flux = np.pad(raw_flux, pad_width=(self.n_pad_model, self.n_pad_model), mode='constant', constant_values=(raw_flux[0], raw_flux[-1]))
        convolved_flux = np.convolve(padded_flux, lsf, 'valid')
        return convolved_flux
    
    def update(self, forward_model, iter_index):
        super().update(forward_model, iter_index)
        
    def init_optimize(self, forward_model):
        lsf_estim = self.build(pars=forward_model.initial_parameters)
        good = np.where(lsf_estim > 1E-10)[0]
        if good.size < lsf_estim.size - 2:
            return
        else:
            f, l = np.min(good), np.max(good)
            nx = l - f + 1
            if nx % 2 == 0:
                nx += 1
                
            self.nx_lsf = nx
            self.compress = int(np.round(self.nx_model * self.nx_lsf))
            self.n_pad_model = int(np.floor(self.nx_lsf / 2))
            self.x = np.arange(-np.floor(self.nx_lsf / 2), np.floor(self.nx_lsf / 2) + 1, 1) * self.dl


class HermiteLSF(LSF):
    """ A Hermite Gaussian LSF model. The model is a sum of Gaussians of constant width and Hermite Polynomial coefficients.

    Attributes:
        hermdeg (int): The degree of the hermite model. Zero corresponds to a pure Gaussian.
    """

    def __init__(self, forward_model, blueprint):

        # Call super
        super().__init__(forward_model, blueprint)

        self.base_par_names = ['_width']

        # The Hermite degree
        self.hermdeg = blueprint['hermdeg']

        for k in range(self.hermdeg):
            self.base_par_names.append('_a' + str(k+1))
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars):
        if self.enabled:
            width = pars[self.par_names[0]].value
            herm = pcmath.hermfun(self.x / width, self.hermdeg)
            if self.hermdeg == 0:  # just a Gaussian
                lsf = herm
            else:
                lsf = herm[:, 0]
            for i in range(self.hermdeg):
                lsf += pars[self.par_names[i+1]].value * herm[:, i+1]
            lsf /= np.nansum(lsf)
            return lsf
        else:
            return self.build_fake()

    def init_parameters(self, forward_model):
        
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[0], value=self.blueprint['width'][1], minv=self.blueprint['width'][0], maxv=self.blueprint['width'][2], mcmcscale=0.1, vary=self.enabled))
        for i in range(self.hermdeg):
            forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i+1], value=self.blueprint['ak'][1], minv=self.blueprint['ak'][0], maxv=self.blueprint['ak'][2], mcmcscale=0.001, vary=True))


class APrioriLSF(LSF):
    """ A container for an LSF known a priori.
    
    Attr:
        default_lsf (np.ndarray): The default LSF model to use.
    """

    def __init__(self, forward_model, blueprint):

        super().__init__(forward_model, blueprint)
        self.base_par_names = []
        self.par_names = []

    def build(self, pars=None):
        return self.default_lsf
    
    def convolve_flux(self, raw_flux, pars=None, lsf=None):
        lsf = build(pars=pars)
        return super().convolve_flux(raw_flux, lsf=lsf)
        


#### Wavelenth Soluton ####

class WavelengthSolution(SpectralComponent):
    """ A base class for a wavelength solution (i.e., conversion from pixels to wavelength).

    Attributes:
        pix_bounds (list): The left and right pixel bounds which correspond to wave_bounds.
        nx (int): The total number of data pixels.
        default_wave_grid (np.ndarray): The default wavelength grid to use or start from.
    """

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        # The pixel bounds which roughly correspond to wave_bounds
        self.pix_bounds = forward_model.pix_bounds

        # The number of total data pixels present in the data
        self.nx = forward_model.n_data_pix
        
        # The default wavelength grid if provided
        if hasattr(forward_model.data, 'default_wave_grid'):
            self.default_wave_grid = forward_model.data.default_wave_grid
        else:
            self.default_wave_grid = None

    # Should never be called. Need to implement if being used
    def build_fake(self):
        raise ValueError('Need to implement a wavelength solution !')

    # Estimates the endpoints of the wavelength grid for each order
    @staticmethod
    def estimate_bounds(forward_model, blueprint):

        if hasattr(forward_model.data, 'default_wave_grid') and forward_model.data.default_wave_grid is not None:
            wave_estimate = np.copy(forward_model.data.default_wave_grid)
            wave_bounds = [wave_estimate[forward_model.pix_bounds[0]], wave_estimate[forward_model.pix_bounds[1]]]
        else:
            # Make an array for the base wavelengths
            quad_wave_set_points = np.array([blueprint['quad_set_point_1'][forward_model.order_num - 1],
                                                     blueprint['quad_set_point_2'][forward_model.order_num - 1],
                                                     blueprint['quad_set_point_3'][forward_model.order_num - 1]])

            # Get the polynomial coeffs through matrix inversion.
            wave_estimate_coeffs = pcmath.poly_coeffs(np.array(blueprint['quad_pixel_set_points']), quad_wave_set_points)

            # The estimated wavelength grid
            wave_estimate = np.polyval(wave_estimate_coeffs, np.arange(forward_model.n_data_pix))

            # Wavelength end points are larger to account for changes in the wavelength solution
            # The stellar template is further padded to account for barycenter sampling
            wave_bounds = [wave_estimate[forward_model.pix_bounds[0]], wave_estimate[forward_model.pix_bounds[1]]]

        return wave_bounds


class PolyWavelengthSolution(WavelengthSolution):
    """ Class for a full wavelength solution defined through cubic splines.

    Attributes:
        poly_order (int): The polynomial order.
        n_splines (int): The number of wavelength splines.
        quad_pixel_set_points (np.ndarray): The three pixel points to use as set points in the quadratic.
        quad_wave_zero_points (np.ndarray): Estimates of the corresonding zero points of quad_pixel_set_points.
        spline_pixel_set_points (np.ndarray): The location of the spline knots in pixel space.
    """

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)
        
        # The polynomial order
        self.poly_order = blueprint['poly_order']
        self.n_poly_pars = self.poly_order + 1
            
        # Parameter names
        self.base_par_names = []
            
        # Required for all instruments to get things going.
        self.quad_pixel_set_points = np.array(blueprint['quad_pixel_set_points'])
        self.quad_wave_zero_points = np.array([blueprint['quad_set_point_1'][self.order_num - 1],
                                               blueprint['quad_set_point_2'][self.order_num - 1],
                                               blueprint['quad_set_point_3'][self.order_num - 1]])
        
        # Estimate the wave grid
        coeffs = pcmath.poly_coeffs(self.quad_pixel_set_points, self.quad_wave_zero_points)
        wave_estimate = np.polyval(coeffs, np.arange(self.nx))
        
        # Polynomial lagrange points
        self.poly_pixel_set_points = np.linspace(self.pix_bounds[0], self.pix_bounds[1], num=self.n_poly_pars).astype(int)
        self.poly_wave_zero_points = wave_estimate[self.poly_pixel_set_points]
        for i in range(self.n_poly_pars):
            self.base_par_names.append('_poly_lagrange_' + str(i + 1))
                
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars):
        
        # The detector grid
        pixel_grid = np.arange(self.nx)
            
        # Lagrange points
        poly_lagrange_pars = np.array([pars[self.par_names[i]].value for i in range(self.n_poly_pars)])
        
        # Get the coefficients
        V = np.vander(self.poly_pixel_set_points, N=self.n_poly_pars)
        Vinv = np.linalg.inv(V)
        coeffs = np.dot(Vinv, self.poly_wave_zero_points + poly_lagrange_pars)
    
        # Build full polynomial
        poly_wave = np.polyval(coeffs, pixel_grid)
        
        return poly_wave

    def init_parameters(self, forward_model):
            
        # Poly parameters
        for i in range(self.n_poly_pars):
            forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint['poly_lagrange'][1], minv=self.blueprint['poly_lagrange'][0], maxv=self.blueprint['poly_lagrange'][2], vary=True))

    def update(self, forward_model, iter_index):
        pass
    

class SplineWavelengthSolution(WavelengthSolution):
    """ Class for a full wavelength solution defined through cubic splines.

    Attributes:
        poly_order (int): The polynomial order.
        n_splines (int): The number of wavelength splines.
        quad_pixel_set_points (np.ndarray): The three pixel points to use as set points in the quadratic.
        quad_wave_zero_points (np.ndarray): Estimates of the corresonding zero points of quad_pixel_set_points.
        spline_pixel_set_points (np.ndarray): The location of the spline knots in pixel space.
    """

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)
        
        # The number of spline knots is n_splines + 1
        self.n_splines = blueprint['n_splines']
        self.n_spline_pars = self.n_splines + 1

        # Parameter names
        self.base_par_names = []
            
        # Required for all instruments to get things going.
        self.quad_pixel_set_points = np.array(blueprint['quad_pixel_set_points'])
        self.quad_wave_zero_points = np.array([blueprint['quad_set_point_1'][self.order_num - 1],
                                               blueprint['quad_set_point_2'][self.order_num - 1],
                                               blueprint['quad_set_point_3'][self.order_num - 1]])
        
        # Estimate the wave grid
        coeffs = pcmath.poly_coeffs(self.quad_pixel_set_points, self.quad_wave_zero_points)
        wave_estimate = np.polyval(coeffs, np.arange(self.nx))

        # Set the spline parameter names and knots
        self.spline_pixel_set_points = np.linspace(self.pix_bounds[0], self.pix_bounds[1], num=self.n_spline_pars).astype(int)
        self.spline_wave_zero_points = wave_estimate[self.spline_pixel_set_points]
        for i in range(self.n_spline_pars):
            self.base_par_names.append('_spline_' + str(i + 1))
                
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars):
        
        # The detector grid
        pixel_grid = np.arange(self.nx)

        # Get the spline parameters
        spline_pars = np.array([pars[self.par_names[i]].value for i in range(self.n_spline_pars)], dtype=np.float64)
        
        # Build the spline model
        spline_wave = scipy.interpolate.CubicSpline(self.spline_pixel_set_points, spline_pars + self.spline_wave_zero_points, extrapolate=False, bc_type='not-a-knot')(pixel_grid)
        
        return spline_wave

    def init_parameters(self, forward_model):

        # Spline parameters
        for i in range(self.n_spline_pars):
            forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], vary=True))

    def update(self, forward_model, iter_index):
        pass


class HybridWavelengthSolution(WavelengthSolution):
    """ Class for a wavelength solution which starts from some pre-derived solution (say a ThAr lamp), with the option of an additional spline offset if further constrained by a gas cell.

    Attributes:
        n_splines (int): The number of wavelength splines.
        splines_enabled (bool): Whether or not the splines are enabled.
        spline_pixel_set_points (np.ndarray): The location of the spline knots.
    """

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        self.base_par_names = []

        # # The number of splines
        self.n_splines = blueprint['n_splines']

        # The number of iterations to delay the wavelength splines
        if self.n_splines == 0:
            self.splines_enabled = False
        else:
            self.splines_enabled = True

        # Construct spline parameter names
        if self.n_splines > 0:
            self.spline_pixel_set_points = np.linspace(self.pix_bounds[0], self.pix_bounds[1], num=self.n_splines + 1)
            for i in range(self.n_splines+1):
                self.base_par_names.append('_wave_spline_' + str(i+1))
        self.par_names = [self.name + s for s in self.base_par_names]
        
        # The known wavelength grid
        if hasattr(forward_model.data, 'default_wave_grid'):
            self.default_wave_grid = forward_model.data.default_wave_grid

    def build(self, pars):
        if not self.splines_enabled:
            return self.default_wave_grid
        else:
            pixel_grid = np.arange(self.nx)
            splines = np.array([pars[self.par_names[i]].value for i in range(self.n_splines + 1)], dtype=np.float64)
            wave_spline = scipy.interpolate.CubicSpline(self.spline_pixel_set_points, splines, bc_type='not-a-knot', extrapolate=False)(pixel_grid)
            return self.default_wave_grid + wave_spline

    def build_fake(self):
        pass

    def init_parameters(self, forward_model):
        
        if self.n_splines > 0:
            for i in range(self.n_splines + 1):
                forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], mcmcscale=0.001, vary=self.splines_enabled))


class LegendreWavelengthSolution(WavelengthSolution):

    """ Class for a full wavelength solution via Hermite Polynomials.

    Attributes:

    """

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        self.base_par_names = []

        # The pixel and wavelength set points for the quadratic (base parameters offset from these lagrange points)
        self.quad_pixel_set_points = np.array(blueprint['quad_pixel_set_points'])
        self.quad_wave_zero_points = np.array([blueprint['quad_set_point_1'][self.order_num - 1],
                                               blueprint['quad_set_point_2'][self.order_num - 1],
                                               blueprint['quad_set_point_3'][self.order_num - 1]])

        # Must be at least 2
        self.legdeg = blueprint['legdeg']

        # Legendre Polynomial Coeffs
        for k in range(self.hermdeg + 1):
            self.base_par_names.append('_a' + str(k))

        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars):

        # The normalized detector grid, [-1, 1] (inclusive)
        pixel_grid = np.linspace(-1, 1, num=self.nx)

        # Construct Legendre polynomials
        wave_sol = np.zeros(self.nx) + pars[self.par_names[0]].value
        for l in range(1, self.leg_order + 1):
            wave_sol += pars[self.par_names[l]].value * legendre(l)(pixel_grid)

        return wave_sol

    def init_parameters(self, forward_model):
        
        # The normalized detector grid, [-1, 1] (inclusive)
        norm_pixel_grid = np.linspace(-1, 1, num=self.nx)
        pfit = pcmath.poly_coeffs((self.quad_pixel_set_points - self.nx / 2) / (self.nx / 2), self.quad_wave_set_points)
        wave = np.polyval(pfit, norm_pixel_grid)
        legfit = np.polynomial.legendre.Legendre.fit(norm_pixel_grid, wave, deg=self.legdeg)
        
        for i in range(self.legdeg + 1):
            forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=legfit[i], minv=legfit[i] - 0.001, maxv=legfit[i] + 0.001, vary=True))

   # To enable/disable splines.
    def update(self, forward_model, iter_index):
        pass


# Misc. models

#### Fringing ####
class FPCavityFringing(EmpiricalMult):
    """ A basic Fabry-Perot cavity model.
    """

    def __init__(self, forward_model, blueprint):

        # Super
        super().__init__(forward_model, blueprint)

        self.base_par_names = ['_d', '_fin']

        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, wave_final):
        if self.enabled:
            d = pars[self.par_names[0]].value
            fin = pars[self.par_names[1]].value
            theta = (2 * np.pi / wave_final) * d
            fringing = 1 / (1 + fin * np.sin(theta / 2)**2)
            return fringing
        else:
            return self.build_fake(wave_final.size)

    def init_parameters(self, forward_model):
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['d'][1], minv=self.blueprint['d'][0], maxv=self.blueprint['d'][2], mcmcscale=0.1, vary=self.enabled))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['fin'][1], minv=self.blueprint['fin'][0], maxv=self.blueprint['fin'][2], mcmcscale=0.1, vary=self.enabled))

    def update(self, forward_model, iter_index):
        super().update(forward_model, iter_index)