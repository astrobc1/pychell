# Science/math
from scipy import constants as cs  # cs.c = speed of light in m/s
import numpy as np  # Math, Arrays
from scipy.special import legendre

# Graphics
import matplotlib.pyplot as plt

# pychell
import pychell.maths as pcmath
from pychell.maths import cspline_interp
import pychell.rvs.template_augmenter as pcaugmenter
import optimparameters.parameters as OptimParameters

# NOTE: The idea is to first define crude "quasi abstract" classes
# Then define a quasi abstract class for each sub type (continuum, star, gas cell, tellurics, etc).
# Then define a concrete class for particular models (SplineContinuum, TAPASTellurics, etc).

#### Quasi Abstract Classes ####
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

    def enable(self, forward_model):
        self.enabled = True
        for pname in self.par_names:
            forward_model.initial_parameters[pname].vary = True
        
    def disable(self, forward_model):
        self.enabled = False
        for pname in self.par_names:
            forward_model.initial_parameters[pname].vary = False

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
        if iter_index + 1 == self.n_delay and not self.enabled:
            self.enabled = True # enable the model
            for pname in self.par_names: # enable the parameters
                forward_model.initial_parameters[pname].vary = True
                
        # Lock any "accidentally" enabled parameters that should be locked
        forward_model.initial_parameters.sanity_lock()

    def __repr__(self):
        """Simple representation method

        Returns:
            str: The string representation of the model.
        """
        return self.name
    
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
    
    def template_yrange(self, forward_model, wave, flux, sregion):
        good = sregion.wave_within(wave)
        max_range = np.max(flux[good]) - np.min(flux[good])
        return max_range
    
    def normalize_template(self, forward_model, wave, flux, uniform=False):
        
        if not uniform:
            dl = np.nanmedian(np.diff(wave))
            wave_min, wave_max = np.nanmin(wave), np.nanmax(wave)
            wave_lin = np.arange(wave_min, wave_max, dl)
            flux_lin = cspline_interp(wave, flux, wave_lin)
        else:
            flux_lin = flux
        
        if 'lsf' in forward_model.models_dict:
            flux_conv = forward_model.models_dict['lsf'].convolve_flux(flux_lin, pars=forward_model.initial_parameters)
        else:
            flux_conv = flux_lin

        data_continuum = pcmath.weighted_median(flux_conv, percentile=0.999)
        flux = flux / data_continuum
        
        return flux
    
    def init_chunk(self, forward_model, templates_dict, sregion, pad=1):
        good = sregion.wave_within(templates_dict[self.key][:, 0], pad=pad)
        templates_dict[self.key] = templates_dict[self.key][good, :]
        templates_dict[self.key][:, 1] = templates_dict[self.key][:, 1] / np.nanmax(templates_dict[self.key][:, 1])

#### Continuum ####
class ContinuumModel(EmpiricalMult):
    
    key = "continuum"
    
    def __init__(self, forward_model, blueprint):
        super().__init__(forward_model, blueprint)
        if "remove" in blueprint:
            self.remove = blueprint["remove"]
        else:
            self.remove = False
    
    @staticmethod
    def estimate_splines(wave, flux, n_splines=6, cont_val=0.75, width=None):
        
        # Number of points
        nx = len(wave)
        
        # Number of knots
        n_knots = n_splines + 1
        
        # Init an array of ones
        continuum_coarse = np.ones(nx, dtype=np.float64)
        
        # Smooth the flux
        flux_smooth = pcmath.median_filter1d(flux, width=7)
        
        # Loop over x and pick out the approximate continuum
        for ix in range(nx):
            use = np.where((wave > wave[ix] - width/2) & (wave < wave[ix] + width/2) & np.isfinite(flux_smooth))[0]
            if use.size == 0 or np.all(~np.isfinite(flux_smooth[use])):
                continuum_coarse[ix] = np.nan
            else:
                continuum_coarse[ix] = pcmath.weighted_median(flux_smooth[use], weights=None, percentile=cont_val)

        good = np.where(np.isfinite(continuum_coarse))[0]
        inds = np.linspace(good[0], good[-1], num=n_knots).astype(int)
        continuum = cspline_interp(wave[inds], continuum_coarse[inds], wave)
        return continuum

    @staticmethod
    def estimate_wobble(wave, flux, mask, poly_order=6, n_sigma=[0.3,3.0], max_iters=50):
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
        
        # Copy the wave and flux
        x = np.copy(wave)
        y = np.copy(flux)
        
        # Smooth the flux
        y = pcmath.median_filter1d(y, 7, preserve_nans=True)
        
        # Create a Vander Matrix to solve
        V = np.vander(x - np.nanmean(x), poly_order + 1)
        
        # Mask to update
        maskcp = np.copy(mask)
        
        # Iteratively solve for continuum
        for i in range(max_iters):
            
            # Solve for continuum
            w = np.linalg.solve(np.dot(V[maskcp].T, V[maskcp]), np.dot(V[maskcp].T, y[maskcp]))
            mu = np.dot(V, w)
            residuals = y - mu
            
            # Effective n sigma
            sigma = np.sqrt(np.nanmedian(residuals**2))
            
            # Update mask
            mask_new = (residuals > -1 * n_sigma[0] * sigma) & (residuals < n_sigma[1] * sigma)
            if maskcp.sum() == mask_new.sum():
                maskcp = mask_new
                break
            else:
                maskcp = mask_new
        return mu
    
    def init_optimize(self, forward_model, templates_dict):
        if self.remove:
            _ = forward_model.init_chunk(templates_dict, forward_model.sregion_order)
            wave = forward_model.models_dict['wavelength_solution'].build(forward_model.initial_parameters)
            continuum_estim = self.fit_continuum_wobble(wave, forward_model.data.flux_chunk, forward_model.data.mask_chunk, poly_order=6, nsigma=[0.3, 3.0], maxniter=50)
            forward_model.data.flux[forward_model.sregion_order.data_inds] /= np.exp(continuum_estim)

class PolyContinuum(ContinuumModel):
    """  Blaze transmission model through a polynomial and/or splines, ideally used after a flat field correction or after remove_continuum but not required.
    
    .. math:
        B(\\lambda) = (\\sum_{k=0}^{N} a_{i} \\lambda^{k} ) sinc(b (\\lambda - \\lambda_{B}))^{2d}
    

    Attributes:
        poly_order (int): The polynomial order.
        n_splines (int): The number of wavelength splines.
        blaze_wave_estimate (bool): The estimate of the blaze wavelegnth. If not provided, defaults to the average of the wavelength grid provided in the build method.
        spline_set_points (np.ndarray): The location of the spline knots.
    """
    
    name = "polynomial_continuum"

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
        poly_cont = np.polyval(poly_pars[::-1], wave_final - self.wave_mid)
        
        return poly_cont

    def init_parameters(self, forward_model):
        
        # Poly parameters
        for i in range(self.n_poly_pars):
            pname = 'poly_' + str(i)
            if pname in self.blueprint:
                forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint[pname][1], minv=self.blueprint[pname][0], maxv=self.blueprint[pname][2], vary=self.enabled))
            else:
                prev = forward_model.initial_parameters[self.par_names[i-1]]
                forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=prev.value/10, minv=prev.minv/10, maxv=prev.maxv/10, vary=self.enabled))
    
    def init_chunk(self, forward_model, templates_dict, sregion):
        self.wave_mid = sregion.midwave()

class SplineContinuum(ContinuumModel):
    """  Blaze transmission model through a polynomial and/or splines, ideally used after a flat field correction or after remove_continuum but not required.
    

    Attributes:
        poly_order (int): The polynomial order.
        n_splines (int): The number of wavelength splines.
        blaze_wave_estimate (bool): The estimate of the blaze wavelegnth. If not provided, defaults to the average of the wavelength grid provided in the build method.
        spline_set_points (np.ndarray): The location of the spline knots.
    """
    
    name = "spline_continuum"

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
        spline_cont = cspline_interp(self.spline_wave_set_points, spline_pars, wave_final)
        
        return spline_cont

    def init_parameters(self, forward_model):
        for ispline in range(self.n_splines + 1):
            forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[ispline], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], vary=self.enabled))
    
    def init_chunk(self, forward_model, templates_dict, sregion):
        wave_estimate = forward_model.models_dict['wavelength_solution'].build(forward_model.initial_parameters)
        good = sregion.wave_within(wave_estimate)
        self.spline_wave_set_points = np.linspace(wave_estimate[good[0]], wave_estimate[good[-1]], num=self.n_splines + 1)

#### Gas Cell ####
class GasCell(TemplateMult):
    """ A gas cell model.
    """
    
    key = "gas_cell"
        
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

class DynamicGasCell(GasCell):
    """ A gas cell model which is consistent across orders.
    """
    
    name = "dynamic_gas_cell"

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        self.base_par_names = ['_shift', '_depth']
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, template, wave_final):
        wave, flux = template[:, 0], template[:, 1]
        wave = wave + pars[self.par_names[0]].value
        flux = flux ** pars[self.par_names[1]].value
        return cspline_interp(wave, flux, wave_final)

    def init_parameters(self, forward_model):
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['shift'][1], minv=self.blueprint['shift'][0], maxv=self.blueprint['shift'][2], vary=True))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['depth'][1], minv=self.blueprint['depth'][0], maxv=self.blueprint['depth'][2], vary=True))
        
class CHIRONGasCell(DynamicGasCell):
    """ A gas cell model which is consistent across orders.
    """
    
    name = "chiron_dynamic_gas_cell"

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        self.base_par_names = ['_shift', '_depth']
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, template, wave_final):
        wave, flux = template[:, 0], template[:, 1]
        wave = wave + pars[self.par_names[0]].value
        flux = flux ** pars[self.par_names[1]].value
        return cspline_interp(wave, flux, wave_final)

    def init_parameters(self, forward_model):
        shift = self.blueprint['shifts'][self.order_num - 1]
        shift_min, shift_max = shift - self.blueprint['shift_range'][0], shift + self.blueprint['shift_range'][1]
        depth = self.blueprint['depth'][1]
        depth_min, depth_max = self.blueprint['depth'][0], self.blueprint['depth'][2]
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=shift, minv=shift_min, maxv=shift_max, vary=True))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=depth, minv=depth_min, maxv=depth_max, vary=True))

class PerfectGasCell(GasCell):
    """ A gas cell model which is consistent across orders.
    """
    
    name = "perfect_gascell"
    
    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        self.par_names = []

    def build(self, pars, template, wave_final):
        wave, flux = template[:, 0], template[:, 1]
        return cspline_interp(wave, flux, wave_final)

#### Star ####
class Star(TemplateMult):
    """ A star model which may or may not have started from a synthetic template.
    
    Attr:
        from_synthetic (bool): Whether or not this model started from a synthetic template or not.
    """
    
    key = "star"
    
    def __init__(self, forward_model, blueprint):
        
        # Super
        super().__init__(forward_model, blueprint)
        
        # Whether or not the star is from a synthetic source
        if "input_file" in blueprint and blueprint["input_file"] is not None:
            self.from_synthetic = True
            self.n_delay = 0
            self.enabled = True
        else:
            self.from_synthetic = False
            self.n_delay = 1
            self.enabled = False
        
        # The augmenter
        if "augmenter" in blueprint:
            self.augmenter = blueprint["augmenter"]
    
    def init_chunk(self, forward_model, templates_dict, sregion):
        pad = 15
        good = sregion.wave_within(templates_dict[self.key][:, 0], pad=pad)
        templates_dict[self.key] = templates_dict[self.key][good, :]
        templates_dict[self.key][:, 1] = templates_dict[self.key][:, 1] / np.nanmax(templates_dict[self.key][:, 1])

class AugmentedStar(Star):
    """ A star model which did not start from a synthetic template.

    Attr:
        from_synthetic (bool): Whether or not this model started from a synthetic template or not.
    """
    
    name = "augmented_star"

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)

        # Pars
        self.base_par_names = ['_vel']
        
        # Update parameter names
        self.par_names = [self.name + s for s in self.base_par_names]
        
        # Augmenter
        if "augmenter" in blueprint and forward_model.n_template_fits > 1:
            self.augmenter = getattr(pcaugmenter, blueprint["augmenter"])
        else:
            self.augmenter = None

    def build(self, pars, template, wave_final):
        wave, flux = template[:, 0], template[:, 1]
        flux_shifted_interp = pcmath.doppler_shift(wave, pars[self.par_names[0]].value, wave_out=wave_final, flux=flux, interp='cspline')
        return flux_shifted_interp

    def init_parameters(self, forward_model):
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=-1*forward_model.data.bc_vel, minv=self.blueprint['vel'][0], maxv=self.blueprint['vel'][2], vary=self.enabled))
        
    def load_template(self, forward_model):
        pad = 15
        wave_uniform = np.arange(forward_model.sregion_order.wavemin - pad, forward_model.sregion_order.wavemax + pad, forward_model.dl)
        if self.from_synthetic:
            print('Loading in Synthetic Stellar Template', flush=True)
            template_raw = np.loadtxt(self.input_file, delimiter=',')
            wave, flux = template_raw[:, 0], template_raw[:, 1]
            flux_interp = cspline_interp(wave, flux, wave_uniform)
            flux_interp /= pcmath.weighted_median(flux_interp, percentile=0.999)
            template = np.array([wave_uniform, flux_interp]).T
        else:
            template = np.array([wave_uniform, np.ones(wave_uniform.size)]).T
        return template
    
    def update_template(self, forward_models, iter_index):
        self.augmenter(forward_models, iter_index)


#### Tellurics ####
class Tellurics(TemplateMult):
    key = "tellurics"
    pass

class TelluricsTAPAS(Tellurics):
    """ A telluric model based on Templates obtained from TAPAS. These templates should be pre-fetched from TAPAS and specific to the observatory. Only water has a unique depth, with all others being identical. The model uses a common Doppler shift.

    Attributes:
        species (list): The names (strings) of the telluric species.
        n_species (int): The number of telluric species.
        species_enabled (dict): A dictionary with species as keys, and boolean values for items (True=enabled, False=disabled)
        species_input_files (list): A list of input files (strings) for the individual species.
    """
    
    name = "tapas_tellurics"

    def __init__(self, forward_model, blueprint):
        super().__init__(forward_model, blueprint)

        self.base_par_names = ['_vel', '_water_depth', '_airmass_depth']
        self.species = ['water', 'methane', 'carbon_dioxide', 'nitrous_oxide', 'oxygen', 'ozone']
        self.species_input_files = blueprint['input_files']
        self.water_enabled, self.airmass_enabled = True, True
        self.thresh = blueprint['flag_thresh']
        self.flag_and_ignore = blueprint['flag_and_ignore']
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, templates, wave_final):
        vel = pars[self.par_names[0]].value
        flux = np.ones(templates[:, 0].size)
        if self.water_enabled:
            flux *= self.build_component(pars, templates, 'water')
        if self.airmass_enabled:
            flux *= self.build_component(pars, templates, 'airmass')
        if vel != 0:
            flux = pcmath.doppler_shift(templates[:, 0], wave_out=wave_final, vel=vel, flux=flux, interp='cspline')
        else:
            flux = cspline_interp(templates[:, 0], flux, wave_final)
        return flux

    def build_component(self, pars, templates, component):
        wave = templates[:, 0]
        if component == 'water':
            depth = pars[self.par_names[1]].value
            flux = templates[:, 1]**depth
        else:
            depth = pars[self.par_names[2]].value
            flux = templates[:, 2]**depth
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
        pad = 5
        
        # Water
        water = np.load(forward_model.templates_path + self.species_input_files['water'])
        wave, flux = water['wave'], water['flux']
        good = np.where((wave > forward_model.sregion_order.wavemin - pad) & (wave < forward_model.sregion_order.wavemax + pad))[0]
        wave_water, flux_water = wave[good], flux[good]
        templates = np.zeros(shape=(wave_water.size, 3), dtype=float)
        templates[:, 0] = wave_water
        templates[:, 1] = flux_water
        
        # Remaining, do in a loop...
        flux_airmass = np.ones(wave_water.size)
        for species in self.species:
            if species == 'water':
                continue
            tell = np.load(forward_model.templates_path + self.species_input_files[species])
            wave, _flux = tell['wave'], tell['flux']
            good = np.where((wave > forward_model.sregion_order.wavemin - pad) & (wave < forward_model.sregion_order.wavemax + pad))[0]
            wave, _flux = wave[good], _flux[good]
            flux_airmass *= cspline_interp(wave, _flux, wave_water)
            
        templates[:, 2] = flux_airmass
            
        return templates
    
    def init_chunk(self, forward_model, templates_dict, sregion):
        
        pad = 1
        good = sregion.wave_within(templates_dict[self.key][:, 0], pad=pad)
        templates_dict[self.key] = templates_dict[self.key][good, :]
        templates_dict[self.key][:, 1] = templates_dict[self.key][:, 1] / np.nanmax(templates_dict[self.key][:, 1])
        templates_dict[self.key][:, 2] = templates_dict[self.key][:, 2] / np.nanmax(templates_dict[self.key][:, 2])
        
        # Check the depth range of the templates
        yrange_water = self.template_yrange(forward_model, templates_dict["tellurics"][:, 0], templates_dict["tellurics"][:, 1], sregion)
        yrange_airmass = self.template_yrange(forward_model, templates_dict["tellurics"][:, 0], templates_dict["tellurics"][:, 2],  sregion)
        if yrange_water > self.flag_thresh[0]:
            self.has_water_features = True
            if self.enabled:
                self.enable(forward_model, "water")
        else:
            self.has_water_features = False
            if self.enabled:
                self.disable(forward_model, "water")
        if yrange_airmass > self.flag_thresh[0]:
            self.has_airmass_features = True
            if self.enabled:
                self.enable(forward_model, "airmass")
        else:
            self.has_airmass_features = False
            if self.enabled:
                self.disable(forward_model, "airmass")
            
        if self.flag_and_ignore:
            self.flag_tellurics(forward_model, templates_dict)
            self.disable(forward_model, "water")
            self.disable(forward_model, "airmass")
    
    def init_optimize(self, forward_model, templates_dict):
        
        # Check the depth range of the templates
        yrange_water = self.template_yrange(forward_model, templates_dict["tellurics"][:, 0], templates_dict["tellurics"][:, 1], forward_model.sregion_order)
        yrange_airmass = self.template_yrange(forward_model, templates_dict["tellurics"][:, 0], templates_dict["tellurics"][:, 2], forward_model.sregion_order)
        
        if yrange_water > self.flag_thresh[0] and self.enabled:
            self.has_water_features = True
            self.enable(forward_model, "water")
        else:
            self.has_water_features = False
            self.disable(forward_model, "water")
        if yrange_airmass > self.flag_thresh[0] and self.enabled:
            self.has_airmass_features = True
            self.enable(forward_model, "airmass")
        else:
            self.has_airmass_features = False
            self.disable(forward_model, "airmass")
            
        # Flag and ignore?
        if self.flag_and_ignore:
            self.flag_tellurics(forward_model, templates_dict)
            self.disable(forward_model, "water")
            self.disable(forward_model, "airmass")
        
        
    def enable(self, forward_model, component):
        if component == "water":
            self.water_enabled = True
            forward_model.initial_parameters[self.par_names[1]].vary = True
        elif component == "airmass":
            self.airmass_enabled = True
            forward_model.initial_parameters[self.par_names[2]].vary = True
        if self.water_enabled or self.airmass_enabled:
            self.enabled = True
            forward_model.initial_parameters[self.par_names[0]].vary = True
            
        
    def disable(self, forward_model, component):
        if component == "water":
            self.water_enabled = False
            forward_model.initial_parameters[self.par_names[1]].vary = False
        elif component == "airmass":
            self.airmass_enabled = False
            forward_model.initial_parameters[self.par_names[2]].vary = False
        if not (self.water_enabled or self.airmass_enabled):
            self.enabled = False
            forward_model.initial_parameters[self.par_names[0]].vary = False
    
    def flag_tellurics(self, forward_model, templates_dict):
        tell_flux_hr = self.build(forward_model.initial_parameters, templates_dict["tellurics"], forward_model.model_wave)
        wave_data = forward_model.models_dict["wavelength_solution"].build(forward_model.initial_parameters)
        tell_flux_lr = cspline_interp(forward_model.model_wave, tell_flux_hr, wave_final)
        tell_flux_lr / pcmath.weighted_median(tell_flux_lr, percentile=0.999)
        bad = np.where(tell_flux_lr < self.thresh[1])[0]
        if bad.size > 0:
            forward_model.data.flux[forward_model.sregion_order.data_inds[bad]] = np.nan
            forward_model.data.flux_unc[forward_model.sregion_order.data_inds[bad]] = np.nan
            forward_model.data.mask[forward_model.sregion_order.data_inds[bad]] = 0

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
    
    key = "lsf"

    def __init__(self, forward_model, blueprint):

        # Call super method
        super().__init__(forward_model, blueprint)
        
        # Set the default LSF if provided
        if hasattr(forward_model.data, 'default_lsf') and forward_model.data.default_lsf is not None:
            self.default_lsf = forward_model.data.default_lsf
        else:
            self.default_lsf = None
            self.dl = forward_model.dl

    # Returns a delta function
    def build_fake(self):
        delta = np.zeros(self.nx, dtype=float)
        delta[int(np.floor(self.nx / 2))] = 1.0
        return delta

    # Convolves the flux
    def convolve_flux(self, raw_flux, pars=None, lsf=None, interp=False):
        if lsf is None and pars is None:
            raise ValueError("Cannot construct LSF with no parameters")
        if not self.enabled:
            return raw_flux
        if lsf is None:
            lsf = self.build(pars)
        convolved_flux = pcmath.convolve_flux(None, raw_flux, R=None, width=None, interp=interp, lsf=lsf, croplsf=False)
        #convolved_flux = pcmath._convolve(raw_flux, lsf)

        return convolved_flux
        
    def init_optimize(self, forward_model, templates_dict):
        pass
            
    def init_chunk(self, forward_model, templates_dict, sregion):
        nx = int((sregion.pix_len() - 1) * forward_model.model_resolution)
        self.dl = forward_model.dl
        x_init = np.arange(int(-nx / 2), int(nx / 2) + 1) * self.dl
        lsf_bad_estim = pcmath.hermfun(x_init / (0.5 * forward_model.initial_parameters[self.par_names[0]].value), deg=0)
        lsf_bad_estim /= np.nanmax(lsf_bad_estim)
        good = np.where(lsf_bad_estim > 1E-10)[0]
        if good.size < lsf_bad_estim.size:
            self.nx = good.size * 2 + 1
        self.x = np.arange(int(-self.nx / 2), int(self.nx / 2) + 1) * self.dl
        self.n_pad_model = int(np.floor(self.nx / 2))

class HermiteLSF(LSF):
    """ A Hermite Gaussian LSF model. The model is a sum of Gaussians of constant width and Hermite Polynomial coefficients.

    Attributes:
        hermdeg (int): The degree of the hermite model. Zero corresponds to a pure Gaussian.
    """
    
    name = "hermite_lsf"

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
    
    name = "modgauss_lsf"

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
    
    name = "apriori_lsf"

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
    
    key = "wavelength_solution"

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
    
    name = "poly_wls"

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
    
    name = "spline_wls"

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

        # Set the spline parameter names and knots
        for i in range(self.n_spline_pars):
            self.base_par_names.append('_spline_' + str(i + 1))
                
        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars):
        
        # The detector grid
        pixel_grid = np.arange(self.sregion.pixmin, self.sregion.pixmax + 1)

        # Get the spline parameters
        spline_pars = np.array([pars[self.par_names[i]].value for i in range(self.n_spline_pars)], dtype=np.float64)
        
        # Build the spline model
        spline_wave = cspline_interp(self.spline_pixel_set_points, spline_pars + self.spline_wave_set_points, pixel_grid)
        
        return spline_wave

    def init_parameters(self, forward_model):

        # Spline parameters
        for i in range(self.n_spline_pars):
            forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], vary=True))

    def update(self, forward_model, iter_index):
        pass
    
    def init_chunk(self, forward_model, templates_dict, sregion):
        self.sregion = sregion
        if self.n_splines > 0:
            wave_estimate = self.estimate_order_wave(forward_model, self.blueprint)
            good = sregion.wave_within(wave_estimate)
            self.nx = sregion.pix_len()
            self.spline_pixel_set_points = np.linspace(good[0], good[-1], num=self.n_splines + 1).astype(int)
            self.spline_wave_set_points = wave_estimate[self.spline_pixel_set_points]

class HybridWavelengthSolution(WavelengthSolution):
    """ Class for a wavelength solution which starts from some pre-derived solution (say a ThAr lamp), with the option of an additional spline offset if further constrained by a gas cell.

    Attributes:
        n_splines (int): The number of wavelength splines.
        splines_enabled (bool): Whether or not the splines are enabled.
        spline_pixel_set_points (np.ndarray): The location of the spline knots.
    """
    
    name = "hybrid_wls"

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
            for i in range(self.n_splines+1):
                self.base_par_names.append('_wave_spline_' + str(i+1))
        self.par_names = [self.name + s for s in self.base_par_names]
        
        # The known wavelength grid
        if hasattr(forward_model.data, 'default_wave_grid'):
            self.default_wave_grid = forward_model.data.default_wave_grid

    def build(self, pars):
        if not self.splines_enabled:
            return self.default_wave_grid[self.sregion.data_inds]
        else:
            pixel_grid = np.arange(self.sregion.pixmin, self.sregion.pixmax + 1)
            splines = np.array([pars[self.par_names[i]].value for i in range(self.n_splines + 1)], dtype=np.float64)
            wave_spline = cspline_interp(self.spline_pixel_set_points, splines, pixel_grid)
            return self.default_wave_grid[self.sregion.data_inds] + wave_spline

    def build_fake(self):
        pass

    def init_parameters(self, forward_model):
        
        if self.n_splines > 0:
            for i in range(self.n_splines + 1):
                forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[i], value=self.blueprint['spline'][1], minv=self.blueprint['spline'][0], maxv=self.blueprint['spline'][2], vary=self.splines_enabled))


    def init_chunk(self, forward_model, templates_dict, sregion):
        self.sregion = sregion
        if self.n_splines > 0:
            wave_estimate = self.estimate_order_wave(forward_model, self.blueprint)
            good = sregion.wave_within(wave_estimate)
            self.spline_pixel_set_points = np.linspace(good[0], good[-1], num=self.n_splines + 1).astype(int)
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
    
    name = "legpoly_wls"

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
    
    name = "fp_fringing"
    key = "fringing"

    def __init__(self, forward_model, blueprint):

        # Super
        super().__init__(forward_model, blueprint)

        self.base_par_names = ['_logd', '_fin']

        self.par_names = [self.name + s for s in self.base_par_names]

    def build(self, pars, wave_final):
        if self.enabled:
            wave_final
            d = np.exp(pars[self.par_names[0]].value)
            fin = pars[self.par_names[1]].value
            theta = (2 * np.pi / wave_final) * d
            fringing = 1 / (1 + fin * np.sin(theta / 2)**2)
            return fringing
        else:
            return self.build_fake(nx)

    def init_parameters(self, forward_model):
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[0], value=self.blueprint['logd'][1], minv=self.blueprint['logd'][0], maxv=self.blueprint['logd'][2], vary=self.enabled))
        forward_model.initial_parameters.add_parameter(OptimParameters.Parameter(name=self.par_names[1], value=self.blueprint['fin'][1], minv=self.blueprint['fin'][0], maxv=self.blueprint['fin'][2], vary=self.enabled))
        
    def init_chunk(self, forward_model, templates_dict, sregion):
        pass