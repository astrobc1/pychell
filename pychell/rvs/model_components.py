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
        
        # Store the blueprint AND auto-populate, may as well...
        self.blueprint = blueprint
        
        # Auto populate self
        for key in blueprint:
            setattr(self, key, blueprint[key])

        # Default enabled, user can further choose to disable after calling super()
        if not hasattr(self, 'n_delay'):
            self.n_delay = 0
        
        # Whether or not to enable this model at the start
        self.enabled = not (self.n_delay > 0)

        # The order number for this model
        self.order_num = forward_model.order_num
        
        # No parameter names, probably overwritten with each instance
        self.base_par_names = []
        self.par_names = []

    # Must implement a build method
    def build(self, pars, *args, **kwargs):
        raise NotImplementedError("Must implement a build method for this Spectral Component")

    # Must implement a build_fake method if ever disabling model
    def build_fake(self, *args, **kwargs):
        raise NotImplementedError("Must implement a build fake method for this Spectral Component")

    # Called after each iteration, may overload.
    def update(self, forward_model, iter_index):
        """Updates this model component given the iteration index and the n_delay attribute. This function may be extended / re-implmented for each model.

        Args:
            forward_model (ForwardModel): The forward model this model belongs to.
            iter_index (int): The iteration index.
        """
        
        # Iteration offset between rvs and general optimizations
        index_offset = 0 if forward_model.models_dict['star'].from_synthetic else 1
        
        # Update based on iteration index
        if (iter_index + 1) == self.n_delay and not self.enabled:
            self.enabled = True
            for pname in self.par_names:
                forward_model.initial_parameters[pname].vary = True
                
        # Lock any "accidentally" enabled parameters
        forward_model.initial_parameters.sanity_lock()

    def __repr__(self):
        """Simple representation method

        Returns:
            str: The string representation of the model.
        """
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'
    
    def init_parameters(self, forward_model):
        """Initializes the parameters for this model

        Args:
            forward_model (ForwardModel): The forward model this model belongs to.
        """
        pass
    
    def init_optimize(self, forward_model, templates_dict):
        """Perform initial, pre-Nelder-Mead optimizations.

        Args:
            forward_model (ForwardModel): The forward model this model belongs to.
        """
        pass
    


class MultModelComponent(SpectralComponent):
    """ Base class for a multiplicative (or log-additive) spectral component.

    Attributes:
        wave_bounds (list): The approximate left and right wavelength endpoints of the considered data.
    """

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
    """ Base class for an empirically derived multiplicative (or log-additive) spectral component (i.e., based purely on parameters, no templates involved). As of now, this is purely a node in the Type heirarchy and provides no unique functionality.
    """
    pass


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
            
            
    def normalize_template(self, forward_model, wave, flux, uniform=False):
        
        if not uniform:
            good = np.where(np.isfinite(wave) & np.isfinite(flux))[0]
            dl = np.nanmedian(np.diff(wave))
            wave_min, wave_max = np.nanmin(wave), np.nanmax(wave)
            wave_lin = np.arange(wave_min, wave_max, dl)
            flux_lin = scipy.interpolate.CubicSpline(wave[good], flux[good], extrapolate=False)(wave_lin)
        else:
            flux_lin = flux
        
        if 'lsf' in forward_model.models_dict:
            flux_conv = forward_model.models_dict['lsf'].convolve_flux(flux_lin, pars=forward_model.initial_parameters)
        else:
            flux_conv = flux_lin

        data_continuum = pcmath.weighted_median(flux_conv, percentile=0.999)
        flux /= data_continuum
        
        return flux
    
    def template_yrange(self, forward_model, wave, flux, sregion):
        good = sregion.wave_within(wave)
        max_range = np.max(flux[good]) - np.min(flux[good])
        return max_range
    
    def init_chunk(self, forward_model, templates_dict, sregion, key):
        pad = 15
        good = sregion.wave_within(templates_dict[key], pad=pad)
        templates_dict[key] = templates_dict[key][good, :]
        templates_dict[key][:, 1] = self.normalize_template(forward_model, templates_dict[key][:, 0], templates_dict[key][:, 1], uniform=False)

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
            
        # Parameter names
        self.base_par_names = []
        for i in range(self.n_poly_pars):
            self.base_par_names.append('_poly_' + str(i)) # starts at zero
                
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, wave_final):
        
        # If not enabled, return ones
        if not self.enabled:
            return self.build_fake(wave_final.size)
        
        # The polynomial coeffs
        poly_pars = np.array([pars[self.par_names[i]].value for i in range(self.poly_order + 1)])
        
        # Build polynomial
        poly_blaze = np.polyval(poly_pars[::-1], wave_final - self.wave_mid)
        
        return poly_blaze

    def init_parameters(self, forward_model):
        
        # Poly parameters
        for i in range(self.n_poly_pars):
            pname = 'poly_' + str(i)
            if pname in self.blueprint:
                forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint[pname][1], minv=self.blueprint[pname][0], maxv=self.blueprint[pname][2], vary=self.enabled))
            else:
                prev = forward_model.initial_parameters[self.par_names[i-1]]
                forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=prev.value/10, minv=prev.minv/10, maxv=prev.maxv/10, vary=self.enabled))

    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'
    
    def init_chunk(self, forward_model, templates_dict, sregion):
        self.wave_mid = sregion.midwave()
    
    
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
        self.spline_wave_set_points = np.linspace(self.wave_bounds[0] - 2, self.wave_bounds[1] + 2, num=self.n_splines + 1)
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
        log_continuum = fit_continuum_wobble(wave, np.log(forward_model.data.flux), forward_model.data.mask, order=4, nsigma=[0.25, 3.0], maxniter=50)
        
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
    
    def init_chunk(self, forward_model, templates_dict, sregion):
        self.wavemid = sregion.midwave()


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
        pad = 5
        template = np.load(self.input_file)
        wave, flux = template['wave'], template['flux']
        good = np.where((wave > forward_model.sregion_order.wavemin - pad) & (wave < forward_model.sregion_order.wavemax + pad))[0]
        wave, flux = wave[good], flux[good]
        flux /= pcmath.weighted_median(flux, percentile=0.999)
        template = np.array([wave, flux]).T
        return template

    def init_parameters(self, forward_model):
        
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['shift'][1], minv=self.blueprint['shift'][0], maxv=self.blueprint['shift'][2], vary=True))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['depth'][1], minv=self.blueprint['depth'][0], maxv=self.blueprint['depth'][2], vary=True))
        
    def init_optimize(self, forward_model, templates_dict):
        wave, flux = templates_dict['gas_cell'][:, 0], templates_dict['gas_cell'][:, 1]
        templates_dict['gas_cell'][:, 1] = self.normalize_template(forward_model, wave, flux, uniform=False)
        
    def init_chunk(self, forward_model, templates_dict, sregion):
        super().init_chunk(forward_model, templates_dict, sregion, "gas_cell")

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
            return pcmath.doppler_shift(wave, pars[self.par_names[0]].value, wave_out=None, flux=flux)
        else:
            return self.build_fake(wave_final.size)

    def init_parameters(self, forward_model):
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=-1*forward_model.data.bc_vel, minv=self.blueprint['vel'][0], maxv=self.blueprint['vel'][2], vary=self.enabled))

    def load_template(self, forward_model):
        pad = 15
        wave_even = np.linspace(forward_model.sregion_order.wavemin - pad, forward_model.sregion_order.wavemax + pad, num=forward_model.n_model_pix_order)
        if self.from_synthetic:
            print('Loading in Synthetic Stellar Template', flush=True)
            template_raw = np.loadtxt(self.input_file, delimiter=',')
            wave, flux = template_raw[:, 0], template_raw[:, 1]
            good = np.where(np.isfinite(wave) & np.isfinite(flux))[0]
            flux_interp = scipy.interpolate.CubicSpline(wave[good], flux[good], extrapolate=False, bc_type='not-a-knot')(wave_even)
            flux_interp /= pcmath.weighted_median(flux_interp, percentile=0.999)
            template = np.array([wave_even, flux_interp]).T
        else:
            template = np.array([wave_even, np.ones(wave_even.size)]).T

        return template
    
    def init_optimize(self, forward_model, templates_dict):
        wave, flux = templates_dict['star'][:, 0], templates_dict['star'][:, 1]
        templates_dict['star'][:, 1] = self.normalize_template(forward_model, wave, flux, uniform=False)
        
    def init_chunk(self, forward_model, templates_dict, sregion):
        super().init_chunk(forward_model, templates_dict, sregion, "star")


#### Tellurics ####
class TelluricsTAPAS(TemplateMult):
    """ A telluric model based on Templates obtained from TAPAS. These templates should be pre-fetched from TAPAS and specific to the observatory. Only water has a unique depth, with all others being identical. The model uses a common Doppler shift.

    Attributes:
        species (list): The names (strings) of the telluric species.
        n_species (int): The number of telluric species.
        species_enabled (dict): A dictionary with species as keys, and boolean values for items (True=enabled, False=disabled)
        species_input_files (list): A list of input files (strings) for the individual species.
    """

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        self.base_par_names = ['_vel', '_water_depth', '_airmass_depth']
        
        self.species = ['water', 'methane', 'carbon_dioxide', 'nitrous_oxide', 'oxygen', 'ozone']
        self.n_species = len(self.species)
        self.species_input_files = blueprint['input_files']

        self.water_enabled, self.airmass_enabled = True, True
        self.min_range = blueprint['min_range']

        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, templates, wave_final):
        if self.enabled:
            vel = pars[self.par_names[0]].value
            flux = np.ones(wave_final.size)
            if self.water_enabled:
                flux *= self.build_component(pars, templates, 'water', wave_final)
            if self.airmass_enabled:
                flux *= self.build_component(pars, templates, 'airmass', wave_final)
            flux = pcmath.doppler_shift(wave_final, vel=vel, flux=flux, interp='linear')
            return flux
        else:
            return self.build_fake(wave_final.size)

    def build_component(self, pars, templates, single_species, wave_final):
        if single_species == 'water':
            depth = pars[self.par_names[1]].value
            wave, flux = templates['water'][:, 0], templates['water'][:, 1]
        else:
            depth = pars[self.par_names[2]].value
            wave, flux = templates['airmass'][:, 0], templates['airmass'][:, 1]
        
        flux = flux ** depth
        return np.interp(wave_final, wave, flux, left=np.nan, right=np.nan)
        return flux

    def init_parameters(self, forward_model):
        
        # Velocity
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['vel'][1], minv=self.blueprint['vel'][0], maxv=self.blueprint['vel'][2], vary=self.enabled))
        
        # Water Depth
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['water_depth'][1], minv=self.blueprint['water_depth'][0], maxv=self.blueprint['water_depth'][2], vary=self.enabled))
        
        # Remaining Components
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[2], value=self.blueprint['airmass_depth'][1], minv=self.blueprint['airmass_depth'][0], maxv=self.blueprint['airmass_depth'][2], vary=self.enabled))

    def load_template(self, forward_model):
        print('Loading in Telluric Templates', flush=True)
        templates = {}
        pad = 5
        
        # Water
        water = np.load(forward_model.templates_path + self.species_input_files['water'])
        wave, flux = water['wave'], water['flux']
        good = np.where((wave > forward_model.sregion_order.wavemin - pad) & (wave < forward_model.sregion_order.wavemax + pad))[0]
        wave, flux = wave[good], flux[good]
        templates['water'] = np.array([wave, flux]).T
        
        # Remaining, do in a loop...
        wave_water = templates['water'][:, 0]
        flux = np.ones(wave_water.size)
        for species in self.species:
            if species == 'water':
                continue
            tell = np.load(forward_model.templates_path + self.species_input_files[species])
            wave, _flux = tell['wave'], tell['flux']
            good = np.where((wave > forward_model.sregion_order.wavemin - pad) & (wave < forward_model.sregion_order.wavemax + pad))[0]
            wave, _flux = wave[good], _flux[good]
            good = np.where(np.isfinite(wave) & np.isfinite(_flux))[0]
            flux *= scipy.interpolate.CubicSpline(wave[good], _flux[good], extrapolate=False)(wave_water)
            
        templates['airmass'] = np.array([wave_water, flux]).T
            
        return templates
    
    def init_chunk(self, forward_model, templates_dict, sregion):
        pad = 2
        for t in templates_dict["tellurics"]:
            super().init_chunk(forward_model, templates_dict["tellurics"], sregion, t)

        yrange_water = self.template_yrange(forward_model, templates_dict["tellurics"]["water"][:, 0], templates_dict["tellurics"]["water"][:, 1], sregion)
        yrange_airmass = self.template_yrange(forward_model, templates_dict["tellurics"]["airmass"][:, 0], templates_dict["tellurics"]["airmass"][:, 1],  sregion)
        if yrange_water > self.min_range:
            self.water_enabled = True
        else:
            self.water_enabled = False
        if yrange_airmass > self.min_range:
            self.airmass_enabled = True
        else:
            self.airmass_enabled = False

    def __repr__(self):
        ss = ' Model Name: ' + self.name
        return ss
    
    def init_optimize(self, forward_model, templates_dict):
        
        # Normalize the water flux
        templates_dict['tellurics']['water'][:, 1] = self.normalize_template(forward_model, templates_dict['tellurics']['water'][:, 0], templates_dict['tellurics']['water'][:, 1], uniform=False)
        
        water_range = self.template_yrange(forward_model, templates_dict['tellurics']['water'][:, 0], templates_dict['tellurics']['water'][:, 1], forward_model.sregion_order)
        
        if water_range > self.min_range:
            self.water_enabled = True
        else:
            self.water_enabled = False
            forward_model.initial_parameters[self.par_names[1]].vary = False
        
        
        # Normalize the other components
        templates_dict['tellurics']['airmass'][:, 1] = self.normalize_template(forward_model, templates_dict['tellurics']['water'][:, 0], templates_dict['tellurics']['airmass'][:, 1], uniform=False)
        
        airmass_range = self.template_yrange(forward_model, templates_dict['tellurics']['airmass'][:, 0], templates_dict['tellurics']['airmass'][:, 1], forward_model.sregion_order)
        if airmass_range > self.min_range:
            self.airmass_enabled = True
        else:
            self.airmass_enabled = False
            forward_model.initial_parameters[self.par_names[2]].vary = False

        # Shift
        shift_vary = True if self.water_enabled or self.airmass_enabled else False
        
        if not shift_vary:
            self.enabled = False
            self.n_delay = int(1E3)
            forward_model.initial_parameters[self.par_names[0]].vary = False


#### LSF ####
class LSF(SpectralComponent):
    """ A base class for an LSF (line spread function) model.

    Attributes:
        dl (float): The step size of the high resolution fidicual wavelength grid the model is convolved on. Must be evenly spaced.
        nx_model (float): The number of model pixels in the high resolution fidicual wavelength grid.
        nx (int): The number of points in the lsf grid.
        x (np.ndarray): The lsf grid.
        default_lsf (np.ndarray): The default LSF to use or start from. Defaults to None.
    """

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)
        
        # Set the default LSF if provided
        if hasattr(forward_model.data, 'default_lsf') and forward_model.data.default_lsf is not None:
            self.default_lsf = forward_model.data.default_lsf
        else:
            self.default_lsf = None
            self.n_model_pix_order = forward_model.n_model_pix_order
            self.nx = np.min([self.n_model_pix_order, 513])
            self.dl = (1 / forward_model.sregion_order.pix_per_wave()) / self.n_model_pix_order
            self.x = np.arange(int(-self.nx / 2), int(self.nx / 2) + 1) * self.dl

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
        convolved_flux = pcmath.convolve_flux(None, raw_flux, R=None, width=None, interp=False, lsf=lsf, croplsf=False)
        return convolved_flux
        
    def init_optimize(self, forward_model, templates_dict):
        pass
            
    def init_chunk(self, forward_model, templates_dict, sregion):
        nx = int((sregion.pix_len() - 1) * forward_model.model_resolution)
        self.dl = (1 / sregion.pix_per_wave()) / forward_model.model_resolution
        x_init = np.arange(int(-nx / 2), int(nx / 2) + 1) * self.dl
        lsf_bad_estim = pcmath.hermfun(x_init / (0.5 * forward_model.initial_parameters[self.par_names[0]].value), deg=0)
        lsf_bad_estim /= np.nanmax(lsf_bad_estim)
        good = np.where(lsf_bad_estim > 1E-10)[0]
        if good.size < lsf_bad_estim.size:
            self.nx = good.size
            if self.nx % 2 == 0:
                self.nx += 1
        self.x = np.arange(int(-self.nx / 2), int(self.nx / 2) + 1) * self.dl
        self.n_pad_model = int(np.floor(self.nx / 2))

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
        
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['width'][1], minv=self.blueprint['width'][0], maxv=self.blueprint['width'][2], vary=self.enabled))
        for i in range(self.hermdeg):
            forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i+1], value=self.blueprint['ak'][1], minv=self.blueprint['ak'][0], maxv=self.blueprint['ak'][2], vary=True))
            
            
class ModGaussLSF(LSF):
    """ A Modified Gaussian LSF model.
    """

    def __init__(self, forward_model, blueprint):

        # Call super
        super().__init__(forward_model, blueprint)

        self.base_par_names = ['_width', '_p']
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars):
        if self.enabled:
            width = pars[self.par_names[0]].value
            p = pars[self.par_names[1]].value
            lsf = np.exp(-0.5 * np.abs(self.x / width)**p)
            lsf /= np.nansum(lsf)
            return lsf
        else:
            return self.build_fake()

    def init_parameters(self, forward_model):
        
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['width'][1], minv=self.blueprint['width'][0], maxv=self.blueprint['width'][2], vary=self.enabled))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['p'][1], minv=self.blueprint['p'][0], maxv=self.blueprint['p'][2], vary=True))


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
    def estimate_order_bounds(forward_model, blueprint):
        wave_estimate = WavelengthSolution.estimate_order_wave(forward_model, blueprint)
        wave_bounds = [wave_estimate[forward_model.crop_data_pix[0]], wave_estimate[forward_model.crop_data_pix[1]]]
        return wave_bounds
    
    @staticmethod
    def estimate_order_wave(forward_model, blueprint):

        if hasattr(forward_model.data, 'default_wave_grid') and forward_model.data.default_wave_grid is not None:
            wave_estimate = np.copy(forward_model.data.default_wave_grid)
        else:
            # Make an array for the base wavelengths
            quad_wave_set_points = np.array([blueprint['quad_set_point_1'][forward_model.order_num - 1],
                                                     blueprint['quad_set_point_2'][forward_model.order_num - 1],
                                                     blueprint['quad_set_point_3'][forward_model.order_num - 1]])

            # Get the polynomial coeffs through matrix inversion.
            wave_estimate_coeffs = pcmath.poly_coeffs(np.array(blueprint['quad_pixel_set_points']), quad_wave_set_points)

            # The estimated wavelength grid
            wave_estimate = np.polyval(wave_estimate_coeffs, np.arange(forward_model.data.flux.size))
            
        return wave_estimate
    
    def init_chunk(self, forward_model, templates_dict, sregion):
        pass


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
        wave_estimate = np.polyval(coeffs, np.arange(forward_model.data.flux.size))
        
        # Polynomial lagrange points
        self.order_poly_pixel_set_points = np.linspace(forward_model.sregion_order.pixmin, forward_model.sregion_order.pixmax, num=self.n_poly_pars).astype(int)
        self.order_poly_wave_zero_points = wave_estimate[self.order_poly_pixel_set_points]
        for i in range(self.n_poly_pars):
            self.base_par_names.append('_poly_lagrange_' + str(i + 1))
                
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars):
        
        # The detector grid
        pixel_grid = np.arange(self.sregion.pixmin, self.sregion.pixmax + 1)
            
        # Lagrange points
        poly_lagrange_pars = np.array([pars[self.par_names[i]].value for i in range(self.n_poly_pars)])
        
        # Get the coefficients
        V = np.vander(self.poly_pixel_set_points, N=self.n_poly_pars)
        Vinv = np.linalg.inv(V)
        coeffs = np.dot(Vinv, self.poly_wave_set_points + poly_lagrange_pars)
    
        # Build full polynomial
        poly_wave = np.polyval(coeffs, pixel_grid)
        
        return poly_wave

    def init_parameters(self, forward_model):
            
        # Poly parameters
        for i in range(self.n_poly_pars):
            forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint['poly_lagrange'][1], minv=self.blueprint['poly_lagrange'][0], maxv=self.blueprint['poly_lagrange'][2], vary=True))

    def update(self, forward_model, iter_index):
        pass
    
    def init_chunk(self, forward_model, templates_dict, sregion):
        wave_estimate = self.estimate_order_wave(forward_model, self.blueprint)
        good = sregion.wave_within(wave_estimate)
        self.sregion = sregion
        self.nx = sregion.pix_len()
        self.poly_pixel_set_points = np.linspace(good[0], good[-1], num=self.poly_order + 1).astype(int)
        self.poly_wave_set_points = wave_estimate[self.poly_pixel_set_points]
        


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
        self.spline_pixel_set_points_order = np.linspace(self.pix_bounds[0], self.pix_bounds[1], num=self.n_spline_pars).astype(int)
        self.spline_wave_zero_points_order = wave_estimate[self.spline_pixel_set_points]
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
    
    def init_chunk(self, forward_model, templates_dict, sregion):
        wave_estimate = self.estimate_order_wave(forward_model, self.blueprint)
        good = sregion.wave_within(wave_estimate)
        self.spline_pixel_set_points = np.linspace(good[0], good[-1], num=self.n_splines + 1)
        self.spline_wave_set_points = wave_estimate[self.spline_pixel_set_points]


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
                forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], vary=self.splines_enabled))


    def init_chunk(self, forward_model, templates_dict, sregion):
        if self.n_splines > 0:
            wave_estimate = self.estimate_order_wave(forward_model, self.blueprint)
            good = sregion.wave_within(wave_estimate)
            self.spline_pixel_set_points = np.linspace(good[0], good[-1], num=self.n_splines + 1)
            self.spline_wave_set_points = wave_estimate[self.spline_pixel_set_points]

class LegPolyWavelengthSolution(WavelengthSolution):
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
        wave_estimate = np.polyval(coeffs, np.arange(forward_model.data.flux.size))
        
        # Polynomial lagrange points
        self.order_poly_pixel_set_points = np.linspace(forward_model.sregion_order.pixmin, forward_model.sregion_order.pixmax, num=self.n_poly_pars).astype(int)
        self.order_poly_wave_zero_points = wave_estimate[self.order_poly_pixel_set_points]
        for i in range(self.n_poly_pars):
            self.base_par_names.append('_poly_lagrange_' + str(i + 1))
                
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars):
        
        # The detector grid
        pixel_grid = np.arange(self.sregion.pixmin, self.sregion.pixmax + 1)
            
        # Lagrange points
        poly_lagrange_pars = np.array([pars[self.par_names[i]].value for i in range(self.n_poly_pars)])
        
        # Get the coefficients
        coeffs = pcmath.leg_coeffs(self.poly_pixel_set_points, self.poly_wave_set_points + poly_lagrange_pars)
    
        # Build full polynomial
        wave_sol = np.zeros(self.nx)
        for i in range(self.n_poly_pars):
            wave_sol += coeffs[i] * scipy.special.eval_legendre(i, pixel_grid)
        
        return wave_sol

    def init_parameters(self, forward_model):
            
        # Poly parameters
        for i in range(self.n_poly_pars):
            forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint['poly_lagrange'][1], minv=self.blueprint['poly_lagrange'][0], maxv=self.blueprint['poly_lagrange'][2], vary=True))

    def init_chunk(self, forward_model, templates_dict, sregion):
        wave_estimate = self.estimate_order_wave(forward_model, self.blueprint)
        good = sregion.wave_within(wave_estimate)
        self.sregion = sregion
        self.nx = sregion.pix_len()
        self.poly_pixel_set_points = np.linspace(good[0], good[-1], num=self.poly_order + 1).astype(int)
        self.poly_wave_set_points = wave_estimate[self.poly_pixel_set_points]

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
            wave_final
            d = pars[self.par_names[0]].value
            fin = pars[self.par_names[1]].value
            theta = (2 * np.pi / wave_final) * d
            fringing = 1 / (1 + fin * np.sin(theta / 2)**2)
            return fringing
        else:
            return self.build_fake(nx)

    def init_parameters(self, forward_model):
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['d'][1], minv=self.blueprint['d'][0], maxv=self.blueprint['d'][2], vary=self.enabled))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['fin'][1], minv=self.blueprint['fin'][0], maxv=self.blueprint['fin'][2], vary=self.enabled))
        
    def init_chunk(self, forward_model, templates_dict, sregion):
        pass
        

# Misc Methods

# This calculates the weighted median of a data set for rolling calculations
def estimate_continuum(x, y, width=7, n_knots=8, cont_val=0.98, smooth=True):
    """This will estimate the continuum with adjustable spline knots.

    Args:
        x (np.ndarray): The wavelength array.
        y (np.ndarray): The flux array.
        width (float, optional): The width of the window in units of x. Defaults to 7.
        n_knots (int, optional): The number of spline knots. Defaults to 8.
        cont_val (float, optional): The estimate of the percentile of the continuum. Defaults to 0.98.
        smooth (bool, optional): Whether or not to smooth the input spectrum. Defaults to True.

    Returns:
        np.ndarray: The estimate of the continuum
    """
    nx = x.size
    continuum_coarse = np.ones(nx, dtype=np.float64)
    if smooth:
        ys = pcmath.median_filter1d(y, width=7)
    else:
        ys = np.copy(y)
    for ix in range(nx):
        use = np.where((x > x[ix]-width/2) & (x < x[ix]+width/2) & np.isfinite(y))[0]
        if use.size == 0 or np.all(~np.isfinite(ys[use])):
            continuum_coarse[ix] = np.nan
        else:
            continuum_coarse[ix] = pcmath.weighted_median(ys[use], weights=None, percentile=cont_val)
    good = np.where(np.isfinite(ys))[0]
    knot_points = x[np.linspace(good[0], good[-1], num=n_knots).astype(int)]
    cspline = scipy.interpolate.CubicSpline(knot_points, continuum_coarse[np.linspace(good[0], good[-1], num=n_knots).astype(int)], extrapolate=False, bc_type='not-a-knot')
    continuum = cspline(x)
    return continuum

def fit_continuum_wobble(x, y, badpix, order=6, nsigma=[0.3,3.0], maxniter=50):
    """Fit the continuum using sigma clipping. This function is nearly identical to Megan Bedell's Wobble code.
    Args:
        x: The wavelengths.
        y: The log-fluxes.
        order: The polynomial order to use
        nsigma: The sigma clipping threshold: tuple (low, high)
        maxniter: The maximum number of iterations to do
    Returns:
        The value of the continuum at the wavelengths in x in log space.
    """
    
    xx = np.copy(x)
    yy = np.copy(y)
    yy = pcmath.median_filter1d(yy, 7, preserve_nans=True)
    A = np.vander(xx - np.nanmean(xx), order+1)
    mask = np.ones(len(xx), dtype=bool)
    badpixcp = np.copy(badpix)
    for i in range(maxniter):
        mask[badpixcp == 0] = 0
        w = np.linalg.solve(np.dot(A[mask].T, A[mask]), np.dot(A[mask].T, yy[mask]))
        mu = np.dot(A, w)
        resid = yy - mu
        sigma = np.sqrt(np.nanmedian(resid**2))
        mask_new = (resid > -nsigma[0]*sigma) & (resid < nsigma[1]*sigma)
        if mask.sum() == mask_new.sum():
            mask = mask_new
            break
        mask = mask_new
    return mu
