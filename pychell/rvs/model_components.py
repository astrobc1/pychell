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
import optimparameters.parameters as OptimParameters


class SpectralComponent:
    """Base class for a general spectral component model.

    Attributes:
        blueprint (dict): The blueprints to construct this component from.
        order_num (int): The image order number.
        enabled (bool): Whether or not this model is enabled.
        name (str): The name of this model, may be anything.
    """

    def __init__(self, blueprint, order_num=None):
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
        if order_num is not None:
            self.order_num = order_num

        # The blueprint must contain a name for this model
        self.name = blueprint['name']

    # Must implement a build method
    def build(self, pars, *args, **kwargs):
        raise NotImplementedError("Must implement a build method for this Spectral Component")

    # Must implement a build_fake method if ever disabling model
    def build_fake(self, *args, **kwargs):
        raise NotImplementedError("Must implement a build fake method for this Spectral Component")

    # Called after each iteration, may overload.
    def update(self, forward_model, iter_num):
        index_offset = 0 if forward_model.models_dict['star'].from_synthetic else 1
        if iter_num + index_offset == self.n_delay and not self.enabled:
            self.enabled = True
            for pname in self.par_names:
                forward_model.initial_parameters[pname].vary = True

    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'

    def init_parameters(self, forward_model):
        self.initial_parameters = OptimParameters.Parameters()
    


class MultModelComponent(SpectralComponent):
    """ Base class for a multiplicative (or log-additive) spectral component.

    Attributes:
        wave_bounds (list): The approximate left and right wavelength endpoints of the considered data.
        base_par_names (list): The base parameter names (constant) for this model.
        par_names (list): The full parameter names for this specific run.
    """

    def __init__(self, blueprint, wave_bounds, order_num=None):

        # Call super method
        super().__init__(blueprint, order_num=order_num)

        # The wavelength bounds (accounting for cropped pixels)
        self.wave_bounds = wave_bounds

    # Effectively no model
    def build_fake(self, nx):
        return np.ones(nx)


class EmpiricalMult(MultModelComponent):
    """ Base class for an empirically derived multiplicative (or log-additive) spectral component (i.e., based purely on parameters, no templates involved).

    """

    def __init__(self, blueprint, wave_bounds, order_num=None):

        # Call super method
        super().__init__(blueprint, wave_bounds, order_num=order_num)

class TemplateMult(MultModelComponent):
    """ A base class for a template based multiplicative model.

    Attributes:
        input_file (str): If provided, stores the full path + filename of the input file.
    """

    def __init__(self, blueprint, wave_bounds, order_num=None):

        # Call super method
        super().__init__(blueprint, wave_bounds, order_num=order_num)

        # By default, set input_file. Some models (like tellurics) ignore this
        if 'input_file' in blueprint:
            self.input_file = blueprint['input_file']


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

    def __init__(self, blueprint, wave_bounds, order_num=None):

        # Super
        super().__init__(blueprint, wave_bounds, order_num=order_num)

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
            blaze_base = np.polyval(
                blaze_base_pars, wave_final - self.blaze_wave_estimate)
        else:
            blaze_base = np.polyval(
                blaze_base_pars, wave_final - np.nanmean(wave_final))

        # If no splines, return the quadratic
        if not self.splines_enabled:
            return blaze_base
        else:
            # Get the spline parameters
            splines = np.array(
                [pars[self.par_names[i+3]].value for i in range(self.n_splines+1)], dtype=np.float64)
            blaze_spline = scipy.interpolate.CubicSpline(
                self.spline_set_points, splines, extrapolate=True, bc_type='not-a-knot')(wave_final)
            return blaze_base + blaze_spline

    # To enable/disable splines.
    def update(self, forward_model, iter_num):
        if iter_num == self.n_delay_splines - 1 and self.n_splines > 0:
            self.splines_enabled = True
            for ispline in range(self.n_splines + 1):
                self.initial_parameters[self.par_names[ispline + 3]].vary = True

    def init_parameters(self, forward_model):
        self.initial_parameters = OptimParameters.Parameters()
        self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['base_quad'][1], minv=self.blueprint['base_quad'][0], maxv=self.blueprint['base_quad'][2], mcmcscale=0.1, vary=self.enabled))
        self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['base_lin'][1], minv=self.blueprint['base_lin'][0], maxv=self.blueprint['base_lin'][2], mcmcscale=0.1, vary=self.enabled))
        self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[2], value=self.blueprint['base_zero'][1], minv=self.blueprint['base_zero'][0], maxv=self.blueprint['base_zero'][2], mcmcscale=0.1, vary=self.enabled))
        if self.n_splines > 0:
            for i in range(self.n_splines + 1):
                self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i+3], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], mcmcscale=0.001, vary=self.splines_enabled))

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

    def __init__(self, blueprint, wave_bounds, order_num=None):

        # Super
        super().__init__(blueprint, wave_bounds, order_num=order_num)

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
        self.initial_parameters = OptimParameters.Parameters()
        self.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[0], value=self.blueprint['base_amp'][1], minv=self.blueprint['base_amp'][0], maxv=self.blueprint['base_amp'][2], mcmcscale=0.1, vary=True))
        self.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[1], value=self.blueprint['base_b'][1], minv=self.blueprint['base_b'][0], maxv=self.blueprint['base_b'][2], mcmcscale=0.1, vary=True))
        self.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[2], value=self.blueprint['base_c'][1], minv=self.blueprint['base_c'][0], maxv=self.blueprint['base_c'][2], mcmcscale=0.1, vary=True))
        self.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[3], value=self.blueprint['base_d'][1], minv=self.blueprint['base_d'][0], maxv=self.blueprint['base_d'][2], mcmcscale=0.1, vary=True))
        if self.n_splines > 0:
            for i in range(self.n_splines + 1):
                self.initial_parameters.add_parameter(OptimParameters.Parameter(
                    name=self.par_names[i+4], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], mcmcscale=0.001, vary=self.splines_enabled))

    # To enable/disable splines.
    def update(self, forward_model, iter_num):
        super().update(forward_model, iter_num)
        if iter_num == self.n_delay_splines - 1 and self.n_splines > 0:
            self.splines_enabled = True
            for ispline in range(self.n_splines + 1):
                self.initial_parameters[self.par_names[ispline + 3]].vary = True

class SplineBlazeModel(EmpiricalMult):
    
    def __init__(self, blueprint, wave_bounds, order_num=None):

        # Super
        super().__init__(blueprint, wave_bounds, order_num=order_num)

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
        splines = np.array([pars[self.par_names[i]].value for i in range(self.n_splines+1)], dtype=np.float64)
        blaze = scipy.interpolate.CubicSpline(self.spline_set_points, splines, extrapolate=True, bc_type='not-a-knot')(wave_final)
        return blaze

    def init_parameters(self, forward_model):
        self.initial_parameters = OptimParameters.Parameters()
        continuum = self.estimate_continuum(forward_model)
        good = np.where(np.isfinite(continuum))[0]
        wave = forward_model.models_dict['wavelength_solution'].build(forward_model.initial_parameters)
        for i in range(self.n_splines + 1):
            v = continuum[pcmath.find_closest(wave[good], self.spline_set_points[i])[0] + good[0]]
            self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=v, minv=v + self.blueprint['spline'][0], maxv=v + self.blueprint['spline'][2], mcmcscale=0.001, vary=self.enabled))

    def update(self, forward_model, iter_num):
        super().update(forward_model, iter_num)
        
        
    def estimate_continuum(self, forward_model):
        n_knots = self.n_splines + 1
        good = np.where(np.isfinite(forward_model.data.flux))[0]
        width = (self.wave_bounds[1] - self.wave_bounds[0]) / (self.n_splines - 1)
        nx = len(forward_model.data.flux)
        continuum_coarse = np.ones(nx, dtype=np.float64)
        ys = pcmath.median_filter1d(forward_model.data.flux, width=7)
        cont_val = 0.98
        wave = forward_model.models_dict['wavelength_solution'].build(forward_model.initial_parameters)
        for ix in range(nx):
            use = np.where((wave > wave[ix]-width/2) & (wave < wave[ix]+width/2) & np.isfinite(ys))[0]
            if use.size == 0 or np.all(~np.isfinite(ys[use])):
                continuum_coarse[ix] = np.nan
            else:
                continuum_coarse[ix] = pcmath.weighted_median(ys[use], med_val=cont_val)
    
        good = np.where(np.isfinite(ys))[0]
        knot_points = wave[good[0::n_knots]]
        cspline = scipy.interpolate.CubicSpline(knot_points, continuum_coarse[good[0::n_knots]], extrapolate=False, bc_type='natural')
        continuum = cspline(wave)
        return continuum


#### Fringing ####

class BasicFringingModel(EmpiricalMult):
    """ A basic Fabry-Perot cavity model.
    """

    def __init__(self, blueprint, wave_bounds, order_num=None):

        # Super
        super().__init__(blueprint, wave_bounds, order_num=order_num)

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
        self.initial_parameters = OptimParameters.Parameters()
        self.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[0], value=self.blueprint['d'][1], minv=self.blueprint['d'][0], maxv=self.blueprint['d'][2], mcmcscale=0.1, vary=self.enabled))
        self.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[1], value=self.blueprint['fin'][1], minv=self.blueprint['fin'][0], maxv=self.blueprint['fin'][2], mcmcscale=0.1, vary=self.enabled))

    def update(self, forward_model, iter_num):
        super().update(forward_model, iter_num)


#### Gas Cell ####

class GasCellModel(TemplateMult):
    """ A gas cell model which is consistent across orders.
    """

    def __init__(self, blueprint, wave_bounds, order_num=None):

        # Call super method
        super().__init__(blueprint, wave_bounds, order_num=order_num)

        self.base_par_names = ['_shift', '_depth']

        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, wave, flux, wave_final):
        if self.enabled:
            # NOTE: Gas shift is additive in Angstroms
            wave = wave + pars[self.par_names[0]].value
            flux = flux ** pars[self.par_names[1]].value
            return np.interp(wave_final, wave, flux, left=flux[0], right=flux[-1])
        else:
            return self.build_fake(wave_final.size)

    def load_template(self, *args, pad=1, **kwargs):
        print('Loading in Gas Cell Template', flush=True)
        template = np.load(self.input_file)
        wave, flux = template['wave'], template['flux']
        good = np.where(
            (wave > self.wave_bounds[0] - pad) & (wave < self.wave_bounds[1] + pad))[0]
        template = np.array([wave[good], flux[good]]).T
        return template

    def init_parameters(self, forward_model):
        self.initial_parameters = OptimParameters.Parameters()
        self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['shift'][1], minv=self.blueprint['shift'][0], maxv=self.blueprint['shift'][2], mcmcscale=0.1, vary=True))
        self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['depth'][1], minv=self.blueprint['depth'][0], maxv=self.blueprint['depth'][2], mcmcscale=0.001, vary=True))


class GasCellModelOrderDependent(TemplateMult):
    """ A gas cell model which is not consistent across orders.
    """

    def __init__(self, blueprint, wave_bounds, order_num=None):

        # Call super method
        super().__init__(blueprint, wave_bounds, order_num=order_num)

        self.base_par_names = ['_shift', '_depth']

        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, wave, flux, wave_final):
        if self.enabled:
            # NOTE: Gas shift is additive in Angstroms
            wave = wave + pars[self.par_names[0]].value
            flux = flux ** pars[self.par_names[1]].value
            return np.interp(wave_final, wave, flux, left=flux[0], right=flux[-1])
        else:
            return self.build_fake(wave_final.size)

    def load_template(self, *args, pad=1, **kwargs):
        print('Loading in Gas Cell Template ...', flush=True)
        template = np.load(self.input_file)
        wave, flux = template['wave'], template['flux']
        good = np.where(
            (wave > self.wave_bounds[0] - pad) & (wave < self.wave_bounds[1] + pad))[0]
        template = np.array([wave[good], flux[good]]).T
        return template

    def init_parameters(self, forward_model):
        self.initial_parameters = OptimParameters.Parameters()
        shift = self.blueprint['shifts'][self.order_num - 1]
        depth = self.blueprint['depth']
        self.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[0], value=shift, minv=shift - self.blueprint['shift_range'][0], maxv=shift + self.blueprint['shift_range'][1], mcmcscale=0.1, vary=True))
        self.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[1], value=depth[1], minv=depth[0], maxv=depth[2], mcmcscale=0.001, vary=True))

#### Star ####

class StarModel(TemplateMult):
    """ A star model which may or may not have started from a synthet template.
    """

    def __init__(self, blueprint, wave_bounds, order_num=None):

        # Call super method
        super().__init__(blueprint, wave_bounds, order_num=order_num)

        self.base_par_names = ['_vel']

        if 'input_file' in blueprint and blueprint['input_file'] is not None:
            self.from_synthetic = True
            self.n_delay = 0
        else:
            self.from_synthetic = False
            self.enabled = False
            self.n_delay = 1

        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, wave, flux, wave_final):
        if self.enabled:
            wave_shifted = wave * np.exp(pars[self.par_names[0]].value / cs.c)
            return np.interp(wave_final, wave_shifted, flux, left=np.nan, right=np.nan)
        else:
            return self.build_fake(wave_final.size)

    def update(self, forward_model, iter_num):
        super().update(forward_model, iter_num)

    def init_parameters(self, forward_model):
        self.initial_parameters = OptimParameters.Parameters()
        self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=-1*forward_model.data.bc_vel, minv=self.blueprint['vel'][0], maxv=self.blueprint['vel'][2], mcmcscale=0.1, vary=self.enabled))

    def load_template(self, nx, pad=15):
        wave_even = np.linspace(
            self.wave_bounds[0] - pad, self.wave_bounds[1] + pad, num=nx)
        if self.from_synthetic:
            print('Loading in Synthetic Stellar Template', flush=True)
            template_raw = np.loadtxt(self.input_file, delimiter=',')
            wave, flux = template_raw[:, 0], template_raw[:, 1]
            flux_interp = scipy.interpolate.CubicSpline(
                wave, flux, extrapolate=False, bc_type='not-a-knot')(wave_even)
            template = np.array([wave_even, flux_interp]).T
        else:
            template = np.array([wave_even, np.ones(wave_even.size)]).T

        return template


class StarModelOrderDependent(TemplateMult):
    """ A star model which is order dependent. For now this is only used for Minerva-Australis.
    """

    def __init__(self, blueprint, wave_bounds, order_num=None):

        # Call super method
        super().__init__(blueprint, wave_bounds, order_num=order_num)

        self.base_par_names = ['_vel']

        if 'input_file' in blueprint:
            self.from_synthetic = True
        else:
            self.from_synthetic = False
            self.enabled = False

        if self.order_num < 10:
            self.ord_str = '0' + str(self.order_num)
        else:
            self.ord_str = str(self.order_num)
        self.input_dir = blueprint['input_dir']
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, wave, flux, wave_final):
        if self.enabled:
            wave_shifted = wave * np.exp(pars[self.par_names[0]].value / cs.c)
            return np.interp(wave_final, wave_shifted, flux, left=np.nan, right=np.nan)
        else:
            return self.build_fake(wave_final.size)

    def update(self, forward_model, iter_num):
        super().update(forward_model, iter_num)

    def init_parameters(self, forward_model):
        self.initial_parameters = OptimParameters.Parameters()
        self.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[0], value=self.blueprint['vel'][1], minv=self.blueprint['vel'][0], maxv=self.blueprint['vel'][2], mcmcscale=0.1, vary=self.enabled))

    def load_template(self, nx, *args, pad=15, **kwargs):
        if self.from_synthetic:
            print('Loading in Synthetic Stellar Template', flush=True)
            f = glob.glob(self.input_dir + '*_' + self.ord_str + '.txt')[0]
            template_raw = np.loadtxt(f, delimiter=',')
            wave_init, flux_init = template_raw[:, 0], template_raw[:, 1]
            good = np.where(
                (wave_init > self.wave_bounds[0] - pad) & (wave_init < self.wave_bounds[1] + pad))[0]
            wave, flux = wave_init[good], flux_init[good]
            wave_star = np.linspace(wave[0], wave[-1], num=nx)
            interp_fun = scipy.interpolate.CubicSpline(
                wave, flux, extrapolate=False, bc_type='not-a-knot')
            flux_star = interp_fun(wave_star)
            flux_star /= pcmath.weighted_median(flux_star, med_val=0.99)
            template = np.array([wave_star, flux_star]).T
        else:
            wave_star = np.linspace(
                self.wave_bounds[0] - pad, self.wave_bounds[1] + pad, num=nx)
            template = np.array([wave_star, np.ones(nx)]).T
        return template


#### Tellurics ####

class TelluricModelTAPAS(TemplateMult):
    """ A telluric model based on Templates obtained from TAPAS. These templates should be pre-fetched from TAPAS and specific to the observatory. Each species has a unique depth, but the model is locked to a common Doppler shift.

    Attributes:
        species (list): The names (strings) of the telluric species.
        n_species (int): The number of telluric species.
        species_enabled (bool): A dictionary with species as keys, and boolean values for items (True=enabled, False=disabled)
        species_input_files (list): A list of input files (strings) for the individual species.
    """

    def __init__(self, blueprint, wave_bounds, order_num=None):

        # Call super method
        super().__init__(blueprint, wave_bounds, order_num=order_num)

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
                self.species_input_files[self.species[itell]
                                         ] = blueprint['species'][self.species[itell]]['input_file']
                self.species_enabled[self.species[itell]] = True
                self.base_par_names.append(
                    '_' + self.species[itell] + '_depth')
        self.par_names = [self.name + s for s in self.base_par_names]

    # Telluric templates is a dictionary of templates.
    # Keys are the telluric names, values are nx * 2 arrays with columns wave, mean_flux

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
        wave, flux = templates[single_species][:,
                                               0], templates[single_species][:, 1]
        flux = flux ** depth
        wave_shifted = wave * np.exp(shift / cs.c)
        return np.interp(wave_final, wave_shifted, flux, left=flux[0], right=flux[-1])

    def build_force(self, pars, templates, wave_final):
        flux = np.ones(wave_final.size, dtype=np.float64)
        for i in range(self.n_species):
            if self.species_enabled[self.species[i]]:
                flux *= self.build_single_species(pars, templates, self.species[i], i, wave_final)
        return flux
        

    def init_parameters(self, forward_model):
        self.initial_parameters = OptimParameters.Parameters()

        # Shift
        self.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[0], value=self.blueprint['vel'][1], minv=self.blueprint['vel'][0], maxv=self.blueprint['vel'][2], mcmcscale=0.1))

        # Components
        for itell, tell in enumerate(self.species):
            max_range = np.nanmax(
                forward_model.templates_dict['tellurics'][tell][:, 1]) - np.nanmin(forward_model.templates_dict['tellurics'][tell][:, 1])
            if max_range > 0.02:
                self.species_enabled[tell] = True
            else:
                self.species_enabled[tell] = False
            self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[itell+1], value=self.blueprint['species'][self.species[itell]]['depth'][1], minv=self.blueprint[
                                                  'species'][self.species[itell]]['depth'][0], maxv=self.blueprint['species'][self.species[itell]]['depth'][2], mcmcscale=0.1, vary=self.species_enabled[tell]))

    def update(self, forward_model, iter_num):
        super().update(forward_model, iter_num)

    def load_template(self, *args, pad=1, **kwargs):
        templates = {}
        for i in range(self.n_species):
            print('Loading in Telluric Template For ' +
                  self.species[i], flush=True)
            template = np.load(self.species_input_files[self.species[i]])
            wave, flux = template['wave'], template['flux']
            good = np.where(
                (wave > self.wave_bounds[0] - pad) & (wave < self.wave_bounds[1] + pad))[0]
            wave, flux = wave[good], flux[good]
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
    """

    def __init__(self, blueprint, dl, nx_model, order_num=None):

        # Call super method
        super().__init__(blueprint, order_num=order_num)

        if 'compress' in blueprint:
            self.compress = blueprint['compress']
        else:
            self.compress = 64

        # The resolution of the model grid (dl = grid spacing)
        self.dl = dl

        # The number of model pixels
        self.nx_model = nx_model

        # The number of points in the grid
        self.nx = int(self.nx_model / self.compress)

        # The actual LSF x grid
        if self.nx % 2 == 0:
            self.x = np.arange(-(int(self.nx / 2) - 1),
                               int(self.nx / 2) + 1, 1) * self.dl
        else:
            self.x = np.arange(-(int(self.nx / 2)),
                               int(self.nx / 2) + 1, 1) * self.dl

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
    
    def update(self, forward_model, iter_num):
        super().update(forward_model, iter_num)


class LSFHermiteModel(LSFModel):
    """ A Hermite Gaussin LSF model. The model is a sum of Gaussians of constant width but coefficients to enforce orthonormality.

    Attributes:
        hermdeg (int): The degree of the hermite model
    """

    def __init__(self, blueprint, dl, nx_model, order_num=None):

        # Call super
        super().__init__(blueprint, dl, nx_model, order_num=order_num)

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
        self.initial_parameters = OptimParameters.Parameters()
        self.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[0], value=self.blueprint['width'][1], minv=self.blueprint['width'][0], maxv=self.blueprint['width'][2], mcmcscale=0.1, vary=self.enabled))
        for i in range(self.hermdeg):
            self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i+1], value=self.blueprint['ak'][1], minv=self.blueprint['ak'][0], maxv=self.blueprint['ak'][2], mcmcscale=0.001, vary=True))


class LSFModelKnown(LSFModel):
    """ A container for a known LSF. In reality this can and probably should not be used and avoided.
    """

    def __init__(self, blueprint, dl, nx_model, order_num=None):

        super().__init__(blueprint, dl, nx_model, order_num=order_num)
        self.base_par_names = []
        self.par_names = []

    def build(self, pars, lsf):
        return lsf

    def init_parameters(self, forward_model):
        self.initial_parameters = OptimParameters.Parameters()


#### Wavelenth Soluton ####

class WavelengthSolutionModel(SpectralComponent):
    """ A base class for a wavelength solution (i.e., conversion from pixels to wavelength).

    Attributes:
        pix_bounds (list): The left and right pixel bounds which correspond to wave_bounds.
        nx (int): The total number of data pixels.
    """

    def __init__(self, blueprint, pix_bounds, nx, order_num=None):

        # Call super method
        super().__init__(blueprint, order_num=order_num)

        # The pixel bounds which roughly correspond to wave_bounds
        self.pix_bounds = pix_bounds

        # The number of total data pixels present in the data
        self.nx = nx

    # Should never be called. Need to implement if being used
    def build_fake(self):
        raise ValueError('Need to implement a wavelength solution !')

    # Estimates the endpoints of the wavelength grid for each order
    @staticmethod
    def estimate_endpoints(data, blueprint, pix_bounds):

        if hasattr(data, 'wave_grid'):
            wave_pad = 1  # Angstroms
            wave_estimate = data.wave_grid  # use the first osbervation
            wave_bounds = [wave_estimate[pix_bounds[0]] - wave_pad, wave_estimate[pix_bounds[1]] + wave_pad]
        else:
            # Make an array for the base wavelengths
            wavesol_base_wave_set_points = np.array([blueprint['base_set_point_1'][data.order_num - 1],
                                                     blueprint['base_set_point_2'][data.order_num - 1],
                                                     blueprint['base_set_point_3'][data.order_num - 1]])

            # Get the polynomial coeffs through matrix inversion.
            wave_estimate_coeffs = pcmath.poly_coeffs(np.array(blueprint['base_pixel_set_points']), wavesol_base_wave_set_points)

            # The estimated wavelength grid
            wave_estimate = np.polyval(wave_estimate_coeffs, np.arange(data.flux.size))

            # Wavelength end points are larger to account for changes in the wavelength solution
            # The stellar template is further padded to account for barycenter sampling
            wave_pad = 1  # Angstroms
            wave_bounds = [wave_estimate[pix_bounds[0]] - wave_pad, wave_estimate[pix_bounds[1]] + wave_pad]

        return wave_bounds


class WaveSolModelSplines(WavelengthSolutionModel):
    """ Class for a full wavelength solution. A "base" solution is provided by a quadratic through set points, and a cubic spline offset is added on to capture any local perturbations.

    Attributes:
        n_splines (int): The number of wavelength splines.
        n_delay_splines (int): The number of iterations to delay the splines.
        splines_enabled (bool): Whether or not the splines are enabled.
        base_pixel_set_points (np.ndarray): The three pixel points to use as set points in the quadratic.
        base_wave_zero_points (np.ndarray): Estimates of the corresonding zero points of base_pixel_set_points.
        spline_pixel_set_points (np.ndarray): The location of the spline knots.
    """

    def __init__(self, blueprint, pix_bounds, nx, order_num=None):

        # Call super method
        super().__init__(blueprint, pix_bounds, nx, order_num=order_num)

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
        self.initial_parameters = OptimParameters.Parameters()
        pfit = pcmath.poly_coeffs(self.base_pixel_set_points, self.base_wave_zero_points)
        wave = np.polyval(pfit, np.arange(self.nx))
        for i in range(self.n_splines + 1):
            self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], mcmcscale=0.001, vary=self.enabled))

    def update(self, forward_model, iter_num):
        pass



class WaveSolModelFull(WavelengthSolutionModel):
    """ Class for a full wavelength solution. A "base" solution is provided by a quadratic through set points, and a cubic spline offset is added on to capture any local perturbations.

    Attributes:
        n_splines (int): The number of wavelength splines.
        n_delay_splines (int): The number of iterations to delay the splines.
        splines_enabled (bool): Whether or not the splines are enabled.
        base_pixel_set_points (np.ndarray): The three pixel points to use as set points in the quadratic.
        base_wave_zero_points (np.ndarray): Estimates of the corresonding zero points of base_pixel_set_points.
        spline_pixel_set_points (np.ndarray): The location of the spline knots.
    """

    def __init__(self, blueprint, pix_bounds, nx, order_num=None):

        # Call super method
        super().__init__(blueprint, pix_bounds, nx, order_num=order_num)

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
        self.initial_parameters = OptimParameters.Parameters()
        self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['base'][1], minv=self.blueprint['base'][0], maxv=self.blueprint['base'][2], mcmcscale=0.1, vary=True))
        self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['base'][1], minv=self.blueprint['base'][0], maxv=self.blueprint['base'][2], mcmcscale=0.1, vary=True))
        self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[2], value=self.blueprint['base'][1], minv=self.blueprint['base'][0], maxv=self.blueprint['base'][2], mcmcscale=0.1, vary=True))
        if self.n_splines > 0:
            for i in range(self.n_splines + 1):
                self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i+3], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], mcmcscale=0.001, vary=self.splines_enabled))

   # To enable/disable splines.
    def update(self, forward_model, iter_num):
        if iter_num == self.n_delay_splines - 1 and self.n_splines > 0:
            self.splines_enabled = True
            for ispline in range(self.n_splines + 1):
                self.initial_parameters[self.par_names[ispline + 3]].vary = True


class WaveModelKnown(WavelengthSolutionModel):

    """ Class for a wavelength solution which starts from some pre-derived solution (say a ThAr lamp).

    Attributes:
        n_splines (int): The number of wavelength splines.
        n_delay_splines (int): The number of iterations to delay the splines.
        splines_enabled (bool): Whether or not the splines are enabled.
        spline_pixel_set_points (np.ndarray): The location of the spline knots.
    """

    def __init__(self, blueprint, pix_bounds, nx, order_num=None):

        # Call super method
        super().__init__(blueprint, pix_bounds, nx, order_num=order_num)

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
            self.spline_pixel_set_points = np.linspace(
                self.pix_bounds[0], self.pix_bounds[1], num=self.n_splines + 1)
            for i in range(self.n_splines+1):
                self.base_par_names.append('_wave_spline_' + str(i+1))
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, wave_grid):
        if not self.splines_enabled:
            return wave_grid
        else:
            pixel_grid = np.arange(self.nx)
            splines = np.array([pars[self.par_names[i]].value for i in range(
                self.n_splines + 1)], dtype=np.float64)
            wave_spline = scipy.interpolate.CubicSpline(
                self.spline_pixel_set_points, splines, bc_type='not-a-knot', extrapolate=True)(pixel_grid)
            return wave_grid + wave_spline

    def build_fake(self):
        pass

    # To enable/disable splines.
    def update(self, forward_model, iter_num):
        if iter_num == self.n_delay_splines - 1 and self.n_splines > 0:
            self.splines_enabled = True
            for ispline in range(self.n_splines + 1):
                self.initial_parameters[self.par_names[ispline + 3]].vary = True

    def init_parameters(self, forward_model):
        self.initial_parameters = OptimParameters.Parameters()
        if self.n_splines > 0:
            for i in range(self.n_splines + 1):
                self.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], mcmcscale=0.001, vary=self.splines_enabled))


class WaveSolModelLegendre(WavelengthSolutionModel):
    """ Class for a full wavelength solution via Legendre Polynomials.

    Attributes:

    """

    def __init__(self, blueprint, wave_bounds, pix_bounds, nx, order_num=None):

        # Call super method
        super().__init__(blueprint, wave_bounds, pix_bounds, nx, order_num=order_num)

        self.base_par_names = []

        # The pixel and wavelength set points for the quadratic (base parameters offset from these lagrange points)
        self.base_pixel_set_points = np.array(
            blueprint['base_pixel_set_points'])
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
        self.initial_parameters = OptimParameters.Parameters()
        self.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[0], value=self.base_wave_zero_points[1], minv=self.base_wave_zero_points[1] - 0.5, maxv=self.base_wave_zero_points[1] + 0.5, mcmcscale=0.001, vary=True))
        self.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[1], value=self.blueprint['coeff1'][1], minv=self.blueprint['coeff1'][0], maxv=self.blueprint['coeff1'][2], mcmcscale=0.001))
        self.initial_parameters.add_parameter(OptimParameters.Parameter(
            name=self.par_names[2], value=self.blueprint['coeff2'][1], minv=self.blueprint['coeff2'][0], maxv=self.blueprint['coeff2'][2], mcmcscale=0.001))

   # To enable/disable splines.
    def update(self, forward_model, iter_num):
        pass
