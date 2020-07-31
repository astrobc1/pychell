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
import scipy.interpolate  # Cubic interpolation, Akima interpolation

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

class ResidualBlazeModel(EmpiricalMult):
    """ Residual blaze transmission model, ideally used after a flat field correction. A quadratic and an additive cubic spline correction are used.

    Attributes:
        n_splines (int): The number of wavelength splines.
        n_delay_splines (int): The number of iterations to delay the splines.
        splines_enabled (bool): Whether or not the splines are enabled.
        blaze_wave_estimate (bool): The estimate of the blaze wavelegnth. If not provided, defaults to the average of the wavelength grid provided in the build method.
        spline_set_points (np.ndarray): The location of the spline knots.
    """

    def __init__(self, forward_model, blueprint):
        
        # Super
        super().__init__(forward_model, blueprint)

        # The base parameter names
        self.base_par_names = ['_base_quad', '_base_lin', '_base_zero']

        # The number of spline knots is n_splines + 1
        self.n_splines = blueprint['n_splines']

        # The number of iterations to delay the wavelength splines
        self.n_delay_splines = blueprint['n_delay_splines']

        # Whether or not the splines are enabled
        if self.n_splines == 0:
            self.splines_enabled = False
        elif self.n_delay_splines > 0:
            self.splines_enabled = False
        else:
            self.splines_enabled = True

        # The estimate of the blaze wavelength
        if 'blaze_wavelengths' in blueprint['blaze_wavelengths']:
            self.blaze_wave_estimate = blueprint['blaze_wavelengths'][self.order_num - 1]
        else:
            self.blaze_wave_estimate = None

        # Set the spline parameter names and knots
        if self.n_splines > 0:
            self.spline_set_points = np.linspace(self.wave_bounds[0], self.wave_bounds[1], num=self.n_splines + 1)
            for i in range(self.n_splines+1):
                self.base_par_names.append('_spline_' + str(i+1))
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, wave_final):

        # The quadratic coeffs
        blaze_base_pars = np.array([pars[self.par_names[0]].value, pars[self.par_names[1]].value, pars[self.par_names[2]].value])

        # Built the quadratic
        if self.blaze_wave_estimate is not None:
            blaze_base = np.polyval(blaze_base_pars, wave_final - self.blaze_wave_estimate)
        else:
            blaze_base = np.polyval(blaze_base_pars, wave_final - np.nanmean(wave_final))

        # If no splines, return the quadratic
        if not self.splines_enabled:
            return blaze_base
        else:
            # Get the spline parameters
            splines = np.array([pars[self.par_names[i+3]].value for i in range(self.n_splines+1)], dtype=np.float64)
            blaze_spline = scipy.interpolate.CubicSpline(self.spline_set_points, splines, extrapolate=True, bc_type='not-a-knot')(wave_final)
            return blaze_base + blaze_spline

    # To enable/disable splines.
    def update(self, forward_model, iter_index):
        if iter_index == self.n_delay_splines - 1 and self.n_splines > 0:
            self.splines_enabled = True
            for ispline in range(self.n_splines + 1):
                forward_model.initial_parameters[self.par_names[ispline + 3]].vary = True

    def init_parameters(self, forward_model):
        
        if forward_model.remove_continuum:
            wave = forward_model.models_dict['wavelength_solution'].build(forward_model.initial_parameters)
            log_continuum = pcaugmenter.fit_continuum_wobble(wave, np.log(forward_model.data.flux), forward_model.data.badpix, order=4, nsigma=[0.3, 3.0], maxniter=50)
            forward_model.data.flux = np.exp(np.log(forward_model.data.flux) - log_continuum)
        
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['base_quad'][1], minv=self.blueprint['base_quad'][0], maxv=self.blueprint['base_quad'][2], mcmcscale=0.1, vary=self.enabled))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['base_lin'][1], minv=self.blueprint['base_lin'][0], maxv=self.blueprint['base_lin'][2], mcmcscale=0.1, vary=self.enabled))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[2], value=self.blueprint['base_zero'][1], minv=self.blueprint['base_zero'][0], maxv=self.blueprint['base_zero'][2], mcmcscale=0.1, vary=self.enabled))
        if self.n_splines > 0:
            for i in range(self.n_splines + 1):
                forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i+3], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], mcmcscale=0.001, vary=self.splines_enabled))

    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ' , Splines Active: ' + str(self.splines_enabled) + ']'


class FullBlazeModel(EmpiricalMult):
    """ A full blaze transmission model. A sinc^(d) and an additive cubic spline correction are used.

    Attributes:
        n_splines (int): The number of wavelength splines.
        n_delay_splines (int): The number of iterations to delay the splines.
        splines_enabled (bool): Whether or not the splines are enabled.
        blaze_wave_estimate (bool): The estimate of the blaze wavelegnth. If not provided, defaults to the average of the wavelength grid provided in the build method.
        spline_set_points (np.ndarray): The location of the spline knots.
    """

    def __init__(self, forward_model, blueprint):

        # Super
        super().__init__(forward_model, blueprint)

        # The base parameter names
        self.base_par_names = ['_base_amp', '_base_b', '_base_c', '_base_d']

        # The number of spline knots is n_splines + 1
        self.n_splines = blueprint['n_splines']

        # The number of iterations to delay the blaze splines
        self.n_delay_splines = blueprint['n_delay_splines']
        if self.n_splines == 0:
            self.splines_enabled = False
        elif self.n_delay_splines > 0:
            self.splines_enabled = False
        else:
            self.splines_enabled = True

        # The estimate of the blaze wavelength
        self.blaze_wave_estimate = blueprint['blaze_wavelengths'][self.order_num - 1]

        # Set the spline parameter names and knots
        if self.n_splines > 0:
            self.spline_set_points = np.linspace(self.wave_bounds[0], self.wave_bounds[1], num=self.n_splines + 1)
            for i in range(self.n_splines+1):
                self.base_par_names.append('_spline_' + str(i+1))
        self.par_names = [self.name + s for s in self.base_par_names]
    
    def build(self, pars, wave_final):
        amp = pars[self.par_names[0]].value
        b = pars[self.par_names[1]].value
        c = pars[self.par_names[2]].value
        d = pars[self.par_names[3]].value
        lam_b = self.blaze_wave_estimate + c
        blaze_base = amp * np.abs(np.sinc(b * (wave_final - lam_b)))**(2 * d)
        if not self.splines_enabled:
            return blaze_base
        else:
            splines = np.array([pars[self.par_names[i+4]].value for i in range(self.n_splines+1)], dtype=np.float64)
            blaze_spline = scipy.interpolate.CubicSpline(
                self.spline_set_points, splines, extrapolate=True, bc_type='not-a-knot')(wave_final)
            return blaze_base + blaze_spline

    def init_parameters(self, forward_model):
        
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['base_amp'][1], minv=self.blueprint['base_amp'][0], maxv=self.blueprint['base_amp'][2], mcmcscale=0.1, vary=True))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['base_b'][1], minv=self.blueprint['base_b'][0], maxv=self.blueprint['base_b'][2], mcmcscale=0.1, vary=True))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[2], value=self.blueprint['base_c'][1], minv=self.blueprint['base_c'][0], maxv=self.blueprint['base_c'][2], mcmcscale=0.1, vary=True))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[3], value=self.blueprint['base_d'][1], minv=self.blueprint['base_d'][0], maxv=self.blueprint['base_d'][2], mcmcscale=0.1, vary=True))
        if self.n_splines > 0:
            for i in range(self.n_splines + 1):
                forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(
                    name=self.par_names[i+4], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], mcmcscale=0.001, vary=self.splines_enabled))

    # To enable/disable splines.
    def update(self, forward_model, iter_index):
        super().update(forward_model, iter_index)
        if iter_index == self.n_delay_splines - 1 and self.n_splines > 0:
            self.splines_enabled = True
            for ispline in range(self.n_splines + 1):
                forward_model.initial_parameters[self.par_names[ispline + 3]].vary = True


class SplineBlazeModel(EmpiricalMult):
    """A general blaze transmission model defined only with cubic splines.

    Attributes:
        n_splines (int): The number of wavelength splines.
        spline_set_points (np.ndarray): The location of the spline knots.
    """
    
    def __init__(self, forward_model, blueprint):

        # Super
        super().__init__(forward_model, blueprint)

        # The base parameter names
        self.base_par_names = []

        # The number of spline knots is n_splines + 1
        self.n_splines = blueprint['n_splines']

        # Set the spline parameters
        self.spline_set_points = np.linspace(self.wave_bounds[0], self.wave_bounds[1], num=self.n_splines + 1)
        for i in range(self.n_splines+1):
            self.base_par_names.append('_spline_' + str(i+1))
        self.par_names = [self.name + s for s in self.base_par_names]
        
    def build(self, pars, wave_final):
        if not self.enabled:
            return self.build_fake(wave_final.size)
        else:
            splines = np.array([pars[self.par_names[i]].value for i in range(self.n_splines+1)], dtype=np.float64)
            blaze = scipy.interpolate.CubicSpline(self.spline_set_points, splines, extrapolate=True, bc_type='not-a-knot')(wave_final)
            return blaze

    def init_parameters(self, forward_model):
        
        if forward_model.remove_continuum:
            wave = forward_model.models_dict['wavelength_solution'].build(forward_model.initial_parameters)
            log_continuum = pcaugmenter.fit_continuum_wobble(wave, np.log(forward_model.data.flux), forward_model.data.badpix, order=4, nsigma=[0.3, 3.0], maxniter=50)
            forward_model.data.flux = np.exp(np.log(forward_model.data.flux) - log_continuum)
            continuum_zero = np.ones_like(forward_model.data.flux)
        else:
            wave = forward_model.models_dict['wavelength_solution'].build(forward_model.initial_parameters)
            log_continuum = pcaugmenter.fit_continuum_wobble(wave, np.log(forward_model.data.flux), forward_model.data.badpix, order=4, nsigma=[0.3, 3.0], maxniter=50)
            continuum_zero = np.exp(log_continuum)
        
        good = np.where(np.isfinite(continuum_zero))[0]
        for i in range(self.n_splines + 1):
            v = continuum_zero[pcmath.find_closest(wave[good], self.spline_set_points[i])[0] + good[0]]
            forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=v, minv=v +self.blueprint['spline'][0], maxv=v + self.blueprint['spline'][1], mcmcscale=0.001, vary=self.enabled))
            
    def update(self, forward_model, iter_index):
        super().update(forward_model, iter_index)
        
        
    def estimate(self, forward_model):
        wave = forward_model.models_dict['wavelength_solution'].estimate(forward_model)
        continuum = pcaugmenter.fit_continuum_wobble(wave, np.log(forward_model.data.flux), forward_model.data.badpix, order=6, nsigma=[0.3, 3.0], maxniter=50)
        return np.exp(continuum)


#### Fringing ####

class BasicFringingModel(EmpiricalMult):
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


#### Gas Cell ####

class GasCellModel(TemplateMult):
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
        good = np.where((wave > self.wave_bounds[0] - pad) & (wave < self.wave_bounds[1] + pad))[0]
        wave, flux = wave[good], flux[good]
        flux /= pcmath.weighted_median(flux, med_val=0.999)
        template = np.array([wave, flux]).T
        return template

    def init_parameters(self, forward_model):
        
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['shift'][1], minv=self.blueprint['shift'][0], maxv=self.blueprint['shift'][2], mcmcscale=0.1, vary=True))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['depth'][1], minv=self.blueprint['depth'][0], maxv=self.blueprint['depth'][2], mcmcscale=0.001, vary=True))


class GasCellModelOrderDependent(TemplateMult):
    """ A gas cell model which is not consistent across orders.
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
        pad = 5
        template = np.load(self.input_file)
        wave, flux = template['wave'], template['flux']
        good = np.where((wave > self.wave_bounds[0] - pad) & (wave < self.wave_bounds[1] + pad))[0]
        wave, flux = wave[good], flux[good]
        flux /= pcmath.weighted_median(flux, med_val=0.999)
        template = np.array([wave, flux]).T
        return template

    def init_parameters(self, forward_model):
        
        shift = self.blueprint['shifts'][self.order_num - 1]
        depth = self.blueprint['depth']
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=shift, minv=shift - self.blueprint['shift_range'][0], maxv=shift + self.blueprint['shift_range'][1], mcmcscale=0.1, vary=True))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=depth[1], minv=depth[0], maxv=depth[2], mcmcscale=0.001, vary=self.enabled))

#### Star ####

class StarModel(TemplateMult):
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
            flux_interp /= pcmath.weighted_median(flux_interp, med_val=0.999)
            template = np.array([wave_even, flux_interp]).T
        else:
            template = np.array([wave_even, np.ones(wave_even.size)]).T

        return template

#### Tellurics ####

class TelluricModelTAPAS(TemplateMult):
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

    def update(self, forward_model, iter_index):
        super().update(forward_model, iter_index)

    def load_template(self, forward_model):
        templates = {}
        pad = 5
        for i in range(self.n_species):
            print('Loading in Telluric Template For ' + self.species[i], flush=True)
            template = np.load(self.species_input_files[self.species[i]])
            wave, flux = template['wave'], template['flux']
            good = np.where((wave > self.wave_bounds[0] - pad) & (wave < self.wave_bounds[1] + pad))[0]
            wave, flux = wave[good], flux[good]
            flux /= pcmath.weighted_median(flux, med_val=0.999)
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


#### LSF ####

class LSFModel(SpectralComponent):
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

        if 'compress' in blueprint:
            self.compress = blueprint['compress']
        else:
            self.compress = 64

        # The resolution of the model grid (dl = grid spacing)
        self.dl = forward_model.dl

        # The number of model pixels
        self.nx_model = forward_model.n_model_pix

        # The number of points in the grid
        self.nx = int(self.nx_model / self.compress)

        # The actual LSF x grid
        if self.nx % 2 == 0:
            self.x = np.arange(-(int(self.nx / 2) - 1),
                               int(self.nx / 2) + 1, 1) * self.dl
        else:
            self.x = np.arange(-(int(self.nx / 2)),
                               int(self.nx / 2) + 1, 1) * self.dl
            
        if hasattr(forward_model.data, 'default_lsf'):
            self.default_lsf = forward_model.data.default_lsf
        else:
            self.default_lsf = None

    # Returns a delta function
    def build_fake(self):
        delta = np.zeros(self.nx, dtype=float)
        delta[int(self.nx / 2)] = 1.0
        return delta

    # Convolves the flux
    def convolve_flux(self, raw_flux, pars=None, lsf=None):
        if lsf is None and pars is None:
            sys.exit("ERROR: Cannot construct LSF with no parameters")
        if not self.enabled:
            return raw_flux 
        if lsf is None:
            lsf = self.build(pars)
        if self.nx % 2 == 0:
            padded_flux = np.pad(raw_flux, pad_width=(int(self.nx / 2 - 1), int(
                self.nx/2)), mode='constant', constant_values=(raw_flux[0], raw_flux[-1]))
        else:
            padded_flux = np.pad(raw_flux, pad_width=(int(
                self.nx / 2), int(self.nx/2)), mode='constant', constant_values=(raw_flux[0], raw_flux[-1]))
        convolved_flux = np.convolve(padded_flux, lsf, 'valid')
        return convolved_flux
    
    def update(self, forward_model, iter_index):
        super().update(forward_model, iter_index)


class LSFHermiteModel(LSFModel):
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


class LSFModelKnown(LSFModel):
    """ A container for a known LSF.
    
    Attr:
        The default LSF model to use.
    """

    def __init__(self, forward_model, blueprint):

        super().__init__(forward_model, blueprint)
        self.base_par_names = []
        self.par_names = []

    def build(self, pars):
        return self.default_lsf
    
    def convolve_flux(self, raw_flux, pars=None, lsf=None):
        return super().convolve_flux(raw_flux, lsf=self.default_lsf)
        


#### Wavelenth Soluton ####

class WavelengthSolutionModel(SpectralComponent):
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
            wave_pad = 1  # Angstroms
            wave_estimate = np.copy(forward_model.data.default_wave_grid)
            wave_bounds = [wave_estimate[forward_model.pix_bounds[0]] - wave_pad, wave_estimate[forward_model.pix_bounds[1]] + wave_pad]
        else:
            # Make an array for the base wavelengths
            wavesol_base_wave_set_points = np.array([blueprint['base_set_point_1'][forward_model.order_num - 1],
                                                     blueprint['base_set_point_2'][forward_model.order_num - 1],
                                                     blueprint['base_set_point_3'][forward_model.order_num - 1]])

            # Get the polynomial coeffs through matrix inversion.
            wave_estimate_coeffs = pcmath.poly_coeffs(np.array(blueprint['base_pixel_set_points']), wavesol_base_wave_set_points)

            # The estimated wavelength grid
            wave_estimate = np.polyval(wave_estimate_coeffs, np.arange(forward_model.n_data_pix))

            # Wavelength end points are larger to account for changes in the wavelength solution
            # The stellar template is further padded to account for barycenter sampling
            wave_pad = 1  # Angstroms
            wave_bounds = [wave_estimate[forward_model.pix_bounds[0]] - wave_pad, wave_estimate[forward_model.pix_bounds[1]] + wave_pad]

        return wave_bounds


class WaveSolModelSplines(WavelengthSolutionModel):
    """ Class for a full wavelength solution.

    Attributes:
        n_splines (int): The number of wavelength splines.
        base_pixel_set_points (np.ndarray): The three pixel points to use as set points in the quadratic.
        base_wave_zero_points (np.ndarray): Estimates of the corresonding zero points of base_pixel_set_points.
        spline_pixel_set_points (np.ndarray): The location of the spline knots in pixel space.
    """

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        self.base_par_names = []

        # The number of wave splines
        self.n_splines = blueprint['n_splines']
        
        self.base_pixel_set_points = np.array(blueprint['base_pixel_set_points'])
        self.base_wave_zero_points = np.array([blueprint['base_set_point_1'][self.order_num - 1],
                                               blueprint['base_set_point_2'][self.order_num - 1],
                                               blueprint['base_set_point_3'][self.order_num - 1]])

        # Spline parameters
        self.spline_pixel_set_points = np.linspace(self.pix_bounds[0], self.pix_bounds[1], num=self.n_splines + 1).astype(int)
        pfit = pcmath.poly_coeffs(self.base_pixel_set_points, self.base_wave_zero_points)
        wave = np.polyval(pfit, np.arange(self.nx))
        self.wave_spline_zero_points = np.array([wave[self.spline_pixel_set_points[i]] for i in range(self.n_splines + 1)], dtype=np.float64)
        for i in range(self.n_splines+1):
            self.base_par_names.append('_wave_spline_' + str(i+1))
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars):
        
        # The detector grid
        pixel_grid = np.arange(self.nx)
        splines = np.array([pars[self.par_names[i]].value for i in range(self.n_splines+1)], dtype=np.float64)
        wave_spline = scipy.interpolate.CubicSpline(self.spline_pixel_set_points, self.wave_spline_zero_points + splines, bc_type='not-a-knot', extrapolate=True)(pixel_grid)
        return wave_spline

    def init_parameters(self, forward_model):
        
        pfit = pcmath.poly_coeffs(self.base_pixel_set_points, self.base_wave_zero_points)
        wave = np.polyval(pfit, np.arange(self.nx))
        for i in range(self.n_splines + 1):
            forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], mcmcscale=0.001, vary=self.enabled))

    def update(self, forward_model, iter_index):
        pass



class WaveSolModelQuadratic(WavelengthSolutionModel):
    """ Class for a full wavelength solution. A "base" solution is provided by a quadratic through set points, and a cubic spline offset is added on to capture any local perturbations.

    Attributes:
        n_splines (int): The number of wavelength splines.
        n_delay_splines (int): The number of iterations to delay the splines.
        splines_enabled (bool): Whether or not the splines are enabled.
        base_pixel_set_points (np.ndarray): The three pixel points to use as set points in the quadratic.
        base_wave_zero_points (np.ndarray): Estimates of the corresonding zero points of base_pixel_set_points.
        spline_pixel_set_points (np.ndarray): The location of the spline knots in pixel space.
    """

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        self.base_par_names = ['_wave_lagrange_1', '_wave_lagrange_2', '_wave_lagrange_3']

        # The number of wave splines
        self.n_splines = blueprint['n_splines']

        # The number of iterations to delay the wavelength splines
        self.n_delay_splines = blueprint['n_delay_splines']
        if self.n_splines == 0:
            self.splines_enabled = False
        elif self.n_delay_splines > 0:
            self.splines_enabled = False
        else:
            self.splines_enabled = True

        # The pixel and wavelength set points for the quadratic (base parameters offset from these lagrange points)
        self.base_pixel_set_points = np.array(blueprint['base_pixel_set_points'])
        self.base_wave_zero_points = np.array([blueprint['base_set_point_1'][self.order_num - 1],
                                               blueprint['base_set_point_2'][self.order_num - 1],
                                               blueprint['base_set_point_3'][self.order_num - 1]])

        # Spline parameters
        if self.n_splines > 0:
            self.spline_pixel_set_points = np.linspace(self.pix_bounds[0], self.pix_bounds[1], num=self.n_splines + 1)
            for i in range(self.n_splines+1):
                self.base_par_names.append('_wave_spline_' + str(i+1))
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars):
        
        # The detector grid
        pixel_grid = np.arange(self.nx)
        base_wave_set_points = np.array([pars[self.par_names[0]].value,
                                         pars[self.par_names[1]].value,
                                         pars[self.par_names[2]].value]) + self.base_wave_zero_points
        
        # The base coefficients
        base_coeffs = pcmath.poly_coeffs(self.base_pixel_set_points, base_wave_set_points)
        wave_base = np.polyval(base_coeffs, pixel_grid)
        
        # Build splines if set
        if not self.splines_enabled:
            return wave_base
        else:
            splines = np.array([pars[self.par_names[i+3]].value for i in range(self.n_splines+1)], dtype=np.float64)
            wave_spline = scipy.interpolate.CubicSpline(self.spline_pixel_set_points, splines, bc_type='not-a-knot', extrapolate=True)(pixel_grid)
            return wave_base + wave_spline

    def init_parameters(self, forward_model):
        
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['base'][1], minv=self.blueprint['base'][0], maxv=self.blueprint['base'][2], mcmcscale=0.1, vary=True))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['base'][1], minv=self.blueprint['base'][0], maxv=self.blueprint['base'][2], mcmcscale=0.1, vary=True))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[2], value=self.blueprint['base'][1], minv=self.blueprint['base'][0], maxv=self.blueprint['base'][2], mcmcscale=0.1, vary=True))
        if self.n_splines > 0:
            for i in range(self.n_splines + 1):
                forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i+3], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], mcmcscale=0.001, vary=self.splines_enabled))

   # To enable/disable splines.
    def update(self, forward_model, iter_index):
        if iter_index == self.n_delay_splines - 1 and self.n_splines > 0:
            self.splines_enabled = True
            for ispline in range(self.n_splines + 1):
                forward_model.initial_parameters[self.par_names[ispline + 3]].vary = True


class WaveModelHybrid(WavelengthSolutionModel):
    """ Class for a wavelength solution which starts from some pre-derived solution (say a ThAr lamp), with the option of an additional spline offset if further constrained by a gas cell.

    Attributes:
        n_splines (int): The number of wavelength splines.
        n_delay_splines (int): The number of iterations to delay the splines.
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
        self.n_delay_splines = blueprint['n_delay_splines']
        if self.n_splines == 0:
            self.splines_enabled = False
        elif self.n_delay_splines > 0:
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
            wave_spline = scipy.interpolate.CubicSpline(self.spline_pixel_set_points, splines, bc_type='not-a-knot', extrapolate=True)(pixel_grid)
            return self.default_wave_grid + wave_spline

    def build_fake(self):
        pass

    # To enable/disable splines.
    def update(self, forward_model, iter_index):
        if iter_index == self.n_delay_splines - 1 and self.n_splines > 0:
            self.splines_enabled = True
            for ispline in range(self.n_splines + 1):
                forward_model.initial_parameters[self.par_names[ispline + 3]].vary = True

    def init_parameters(self, forward_model):
        
        if self.n_splines > 0:
            for i in range(self.n_splines + 1):
                forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], mcmcscale=0.001, vary=self.splines_enabled))



class WaveSolModelLegendre(WavelengthSolutionModel):

    """ Class for a full wavelength solution via Legendre Polynomials. In development.

    Attributes:

    """

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        self.base_par_names = []

        # The pixel and wavelength set points for the quadratic (base parameters offset from these lagrange points)
        self.base_pixel_set_points = np.array(blueprint['base_pixel_set_points'])
        self.base_wave_zero_points = np.array([blueprint['base_set_point_1'][self.order_num - 1],
                                               blueprint['base_set_point_2'][self.order_num - 1], blueprint['base_set_point_3'][self.order_num - 1]])

        self.leg_order = blueprint['leg_order']

        # Legendre Polynomial Coeffss
        for i in range(self.leg_order + 1):
            self.base_par_names.append('_leg_coeff_' + str(i))

        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars):

        # The normalized detector grid, [-1, 1] (inclusive)
        pixel_grid = np.linspace(-self.nx, self.nx, num=self.nx) / self.nx

        # Construct Legendre polynomials
        wave_sol = np.zeros(self.nx) + pars[self.par_names[0]].value
        for l in range(1, self.leg_order + 1):
            wave_sol += pars[self.par_names[l]].value * legendre(l, monic=False)(pixel_grid)

        return wave_sol

    def init_parameters(self, forward_model):
        
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[0], value=self.base_wave_zero_points[1], minv=self.base_wave_zero_points[1] - 0.5, maxv=self.base_wave_zero_points[1] + 0.5, mcmcscale=0.001, vary=True))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[1], value=self.blueprint['coeff1'][1], minv=self.blueprint['coeff1'][0], maxv=self.blueprint['coeff1'][2], mcmcscale=0.001))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[2], value=self.blueprint['coeff2'][1], minv=self.blueprint['coeff2'][0], maxv=self.blueprint['coeff2'][2], mcmcscale=0.001))

   # To enable/disable splines.
    def update(self, forward_model, iter_index):
        pass
