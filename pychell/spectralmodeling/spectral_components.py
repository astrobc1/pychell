# Maths
import numpy as np
from scipy.special import eval_legendre

# pychell
import pychell.maths as pcmath

# Optimize
from optimize.models import Model
from optimize.knowledge import BoundedParameters, BoundedParameter


####################
#### BASE TYPES ####
####################

class SpectralComponent(Model):
    """Base class for a general spectral component model.

    Attributes:
        blueprint (dict): The blueprints to construct this component from.
        order_num (int): The image order number.
        enabled (bool): Whether or not this model is enabled.
        n_delay (int): The number of iterations to delay this model component.
        base_par_names (str): The base parameter names of the parameters for this model.
        par_names (str): The full parameter names are name + _ + base_par_names.
    """

    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, blueprint, sregion, order_num):
        """ Base initialization for a model component.

        Args:
            blueprint (dict): The blueprints to construct this component from.
            order_num (int): The image order number.
        """
        
        # The order number for this model
        self.order_num = order_num
        
        # The spectral region
        self.sregion = sregion
        
        # Store the blueprint
        self.blueprint = blueprint

        # No parameter names, probably overwritten with each instance
        self.base_par_names = []
        self.par_names = []
        
    def _init_parameters(self, data):
        return BoundedParameters()

    ####################
    #### INITIALIZE ####
    ####################

    def initialize(self, p0, data, templates_dict, iter_index=None):
        """Initializes this model, does nothing.
        """
        pass

    def lock_pars(self, pars):
        for pname in self.par_names:
            pars[pname].vary = False

    ###############
    #### MISC. ####
    ###############

    def vary_pars(self, pars):
        for pname in self.par_names:
            pars[pname].vary = True

    def __repr__(self):
        s = f"Spectral model: {self.name}\n"
        s += "Parameters:"
        for pname in self.par_names:
            s += f"{pname}\n"
        return s

class MultModelComponent(SpectralComponent):
    """ Base class for a multiplicative (or log-additive) spectral component.

    Attributes:
        wave_bounds (list): The approximate left and right wavelength endpoints of the considered data.
    """
    pass

class EmpiricalMult(MultModelComponent):
    """ Base class for an empirically derived multiplicative (or log-additive) spectral component (i.e., based purely on parameters, no templates involved). As of now, this is purely a node in the Type heirarchy and provides no unique functionality.
    """
    pass

class TemplateMult(MultModelComponent):
    """ A base class for a template based multiplicative model.

    Attributes:
        input_file (str): If provided, stores the full path + filename of the input file.
    """
    
    #############################
    #### CONSTRUCTOR HELPERS ####
    #############################
    
    def _init_templates(self, *args, **kwargs):
        pass


    ###############
    #### MISC. ####
    ###############
    
    def get_template_yrange(self, wave, flux, sregion):
        good = sregion.wave_within(wave)
        max_range = np.nanmax(flux[good]) - np.nanmin(flux[good])
        return max_range


##########################
#### CONTINUUM MODELS ####
##########################

class ContinuumModel(EmpiricalMult):
    
    key = "continuum"
    
    ###############
    #### MISC. ####
    ###############
    
    @staticmethod
    def estimate_wobble(wave, flux, mask, poly_order=4, n_sigma=(0.3,3.0), max_iters=50):
        """Fit the continuum using sigma clipping. This function is nearly identical to Megan Bedell's Wobble code.
        Args:
            x (np.ndarray): The wavelengths.
            y (np.ndarray): The fluxes.
            poly_order (int): The polynomial order to use
            n_sigma (tuple): The sigma clipping threshold: (low, high)
            ma_iters: The maximum number of iterations to do.
        Returns:
            The value of the continuum at the wavelengths in x in log space.
        """
        
        # Copy the wave and flux
        x = np.copy(wave)
        y = np.copy(flux)
        
        # Smooth the flux first
        y = pcmath.median_filter1d(y, 7, preserve_nans=True)
        
        # Create a Vander Matrix to solve
        V = np.vander(x - np.nanmean(x), poly_order + 1)
        
        # Mask to update
        maskcp = np.copy(mask)
        
        # Iteratively solve for continuum
        for i in range(max_iters):
            
            # Solve for continuum
            w = np.linalg.solve(np.dot(V[maskcp].T, V[maskcp]), np.dot(V[maskcp].T, y[maskcp]))
            cont = np.dot(V, w)
            residuals = y - cont
            
            # Effective n sigma
            sigma = np.sqrt(np.nanmedian(residuals**2))
            
            # Update mask
            mask_new = (residuals > -1 * n_sigma[0] * sigma) & (residuals < n_sigma[1] * sigma)
            if maskcp.sum() == mask_new.sum():
                maskcp = mask_new
                break
            else:
                maskcp = mask_new
        return cont

class PolyContinuum(ContinuumModel):
    """Blaze transmission model through a polynomial.
    """
    
    name = "polynomial_continuum"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, blueprint, sregion, order_num):
        
        # Super
        super().__init__(blueprint, sregion, order_num)
        
        # The polynomial order
        self.poly_order = self.blueprint['poly_order']
        self.n_poly_pars = self.poly_order + 1
        
        # The middle of the order
        self.wave_mid = self.sregion.midwave()
            
        # Parameter names
        self.base_par_names = []
        for i in range(self.n_poly_pars):
            self.base_par_names.append(f"_poly_{i}")
                
        self.par_names = [self.name + s for s in self.base_par_names]

    def _init_parameters(self, data):
        
        # Parameters
        pars = BoundedParameters()
        
        # Poly parameters
        for i in range(self.n_poly_pars):
            pname = f"poly_{i}"
            if pname in self.blueprint:
                pars[self.par_names[i]] = BoundedParameter(value=self.blueprint[pname][1],
                                                        vary=True,
                                                        lower_bound=self.blueprint[pname][0], upper_bound=self.blueprint[pname][2])
            else:
                prev = pars[self.par_names[i - 1]]
                pars[self.par_names[i]] = BoundedParameter(value=prev.value/10,
                                                        vary=True,
                                                        lower_bound=prev.lower_bound/10, upper_bound=prev.upper_bound/10)
        return pars
    
    
    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars, wave_final):
        
        # The polynomial coeffs
        poly_pars = np.array([pars[self.par_names[i]].value for i in range(self.poly_order + 1)])
        
        # Build polynomial
        poly_cont = np.polyval(poly_pars[::-1], wave_final - self.wave_mid)
        
        return poly_cont

class SplineContinuum(ContinuumModel):
    """  Blaze transmission model through a polynomial and/or splines, ideally used after a flat field correction or after remove_continuum but not required.
    

    Attributes:
        poly_order (int): The polynomial order.
        n_splines (int): The number of wavelength splines.
        blaze_wave_estimate (bool): The estimate of the blaze wavelegnth. If not provided, defaults to the average of the wavelength grid provided in the build method.
        spline_set_points (np.ndarray): The location of the spline knots.
    """
    
    name = "spline_continuum"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, blueprint, sregion, order_num):
        
        # Super
        super().__init__(blueprint, sregion, order_num)

        # The number of spline knots is n_splines + 1
        self.n_splines = self.blueprint['n_splines']

        # The wavelength zero points for each knot
        self.spline_wave_set_points = np.linspace(self.sregion.wavemin, self.sregion.wavemax, num=self.n_splines + 1)

        # Set the spline parameter names and knots
        for i in range(self.n_splines+1):
            self.base_par_names.append('_spline_' + str(i+1))

        self.par_names = [self.name + s for s in self.base_par_names]


    def _init_parameters(self, data):
        pars = BoundedParameters()
        for ispline in range(self.n_splines + 1):
            pars[self.par_names[ispline]] = BoundedParameter(value=self.blueprint['spline_lagrange'][1],
                                                             vary=True,
                                                             lower_bound=self.blueprint['spline_lagrange'][0],
                                                             upper_bound=self.blueprint['spline_lagrange'][2])
        return pars
    
    
    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars, wave_final):

        # Get the spline parameters
        spline_pars = np.array([pars[self.par_names[i]].value for i in range(self.n_splines + 1)], dtype=np.float64)

        # Build
        spline_cont = pcmath.cspline_interp(self.spline_wave_set_points, spline_pars, wave_final)
        
        return spline_cont


#########################
#### GAS CELL MODELS ####
#########################

class GasCell(TemplateMult):
    """ A gas cell model.
    """
    
    key = "gas_cell"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, blueprint, sregion, order_num):

        # Call super method
        super().__init__(blueprint, sregion, order_num)
        
        self.input_file = self.blueprint["input_file"]
        
    def _init_templates(self, data, templates_path, model_dl):
        print('Loading Gas Cell Template', flush=True)
        pad = 5
        template = np.load(templates_path + self.input_file)
        wave, flux = template['wave'], template['flux']
        good = np.where((wave > self.sregion.wavemin - pad) & (wave < self.sregion.wavemax + pad))[0]
        wave, flux = wave[good], flux[good]
        flux /= pcmath.weighted_median(flux, percentile=0.999)
        template = np.array([wave, flux]).T
        return template

class DynamicGasCell(GasCell):
    """ A gas cell model which is consistent across orders.
    """
    
    name = "dynamic_gas_cell"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, blueprint, sregion, order_num):

        # Call super method
        super().__init__(blueprint, sregion, order_num)

        self.base_par_names = ['_shift', '_depth']
        self.par_names = [self.name + s for s in self.base_par_names]

    def _init_parameters(self, data):
        pars = BoundedParameters()
        pars[self.par_names[0]] = BoundedParameter(value=self.blueprint['shift'][1],
                                                       vary=True,
                                                       lower_bound=self.blueprint['shift'][0],
                                                       upper_bound=self.blueprint['shift'][2])
        
        pars[self.par_names[1]] = BoundedParameter(value=self.blueprint['depth'][1],
                                                       vary=True,
                                                       lower_bound=self.blueprint['depth'][0],
                                                       upper_bound=self.blueprint['depth'][2])
        
        return pars


    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars, template, wave_final):
        wave, flux = template[:, 0], template[:, 1]
        wave = wave + pars[self.par_names[0]].value
        flux = flux ** pars[self.par_names[1]].value
        return pcmath.cspline_interp(wave, flux, wave_final)

class PerfectGasCell(GasCell):
    """ A gas cell model which is consistent across orders.
    """
    
    name = "perfect_gascell"

    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars, template, wave_final):
        wave, flux = template[:, 0], template[:, 1]
        return pcmath.cspline_interp(wave, flux, wave_final)


#####################
#### STAR MODELS ####
#####################

class Star(TemplateMult):
    """ A star model which may or may not have started from a synthetic template.
    
    Attr:
        from_synthetic (bool): Whether or not this model started from a synthetic template or not.
    """
    key = "star"

class AugmentedStar(Star):
    """ A star model which did not start from a synthetic template.

    Attr:
        from_synthetic (bool): Whether or not this model started from a synthetic template or not.
    """
    
    name = "augmented_star"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, blueprint, sregion, order_num):

        # Call super method
        super().__init__(blueprint, sregion, order_num)
        
        # Whether or not the star is from a synthetic source
        if "input_file" in self.blueprint and self.blueprint["input_file"] is not None:
            self.from_flat = False
            self.input_file = self.blueprint["input_file"]
        else:
            self.from_flat = True

        # Pars
        self.base_par_names = ['_vel']
        
        # Update parameter names
        self.par_names = [self.name + s for s in self.base_par_names]

    def _init_parameters(self, data):
        pars = BoundedParameters()
        pars[self.par_names[0]] = BoundedParameter(value=self.blueprint['vel'][1],
                                                       vary=True,
                                                       lower_bound=self.blueprint['vel'][0],
                                                       upper_bound=self.blueprint['vel'][2])
    
        return pars
        
    def _init_templates(self, data, templates_path, model_dl):
        pad = 15
        wave_uniform = np.arange(self.sregion.wavemin - pad, self.sregion.wavemax + pad, model_dl)
        if not self.from_flat:
            print("Loading Stellar Template", flush=True)
            template_raw = np.loadtxt(templates_path + self.input_file, delimiter=',')
            wave, flux = template_raw[:, 0], template_raw[:, 1]
            flux_interp = pcmath.cspline_interp(wave, flux, wave_uniform)
            flux_interp /= pcmath.weighted_median(flux_interp, percentile=0.999)
            template = np.array([wave_uniform, flux_interp]).T
            self.initial_template = np.copy(template)
        else:
            template = np.array([wave_uniform, np.ones_like(wave_uniform)]).T
            self.initial_template = np.copy(template)
        return template
        
    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars, template, wave_final):
        wave, flux = template[:, 0], template[:, 1]
        flux = pcmath.doppler_shift(wave, pars[self.par_names[0]].value, wave_out=wave_final, flux=flux, interp='cspline')
        return flux


    ####################
    #### INITIALIZE ####
    ####################
    
    def initialize(self, p0, data, templates_dict, iter_index=None):
        if iter_index == 0 and self.from_flat:
            p0[self.par_names[0]].vary = False


#########################
#### TELLURIC MODELS ####
#########################

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
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, blueprint, sregion, order_num):
        super().__init__(blueprint, sregion, order_num)

        self.base_par_names = ['_vel', '_water_depth', '_airmass_depth']
        self.species = ['water', 'methane', 'carbon_dioxide', 'nitrous_oxide', 'oxygen', 'ozone']
        self.species_input_files = self.blueprint['input_files']
        self.has_water_features, self.has_airmass_features = True, True
        self.feature_depth = self.blueprint['feature_depth']
        self.par_names = [self.name + s for s in self.base_par_names]

    def _init_parameters(self, data):
        
        pars = BoundedParameters()
        
        # Velocity
        pars[self.par_names[0]] = BoundedParameter(value=self.blueprint['vel'][1],
                                                       vary=(self.has_water_features or self.has_airmass_features),
                                                       lower_bound=self.blueprint['vel'][0],
                                                       upper_bound=self.blueprint['vel'][2])
        
        # Water Depth
        pars[self.par_names[1]] = BoundedParameter(value=self.blueprint['water_depth'][1],
                                                       vary=self.has_water_features,
                                                       lower_bound=self.blueprint['water_depth'][0],
                                                       upper_bound=self.blueprint['water_depth'][2])
        
        # Remaining Components
        pars[self.par_names[2]] = BoundedParameter(value=self.blueprint['airmass_depth'][1],
                                                       vary=self.has_airmass_features,
                                                       lower_bound=self.blueprint['airmass_depth'][0],
                                                       upper_bound=self.blueprint['airmass_depth'][2])
        
        return pars

    def _init_templates(self, data, templates_path, model_dl):
        print('Loading Telluric Templates', flush=True)
        
        # Pad
        pad = 5
        
        # Water
        water = np.load(templates_path + self.species_input_files['water'])
        wave, flux = water['wave'], water['flux']
        good = np.where((wave > self.sregion.wavemin - pad) & (wave < self.sregion.wavemax + pad))[0]
        wave_water, flux_water = wave[good], flux[good]
        templates = np.zeros(shape=(wave_water.size, 3), dtype=float)
        templates[:, 0] = wave_water
        if np.nanmax(flux_water) - np.nanmin(flux_water) < self.feature_depth:
            self.has_water_features = False
        templates[:, 1] = flux_water
        
        # Remaining, do in a loop...
        flux_airmass = np.ones(wave_water.size)
        for species in self.species:
            if species == 'water':
                continue
            tell = np.load(templates_path + self.species_input_files[species])
            wave, _flux = tell['wave'], tell['flux']
            good = np.where((wave > self.sregion.wavemin - pad) & (wave < self.sregion.wavemax + pad))[0]
            wave, _flux = wave[good], _flux[good]
            flux_airmass *= pcmath.cspline_interp(wave, _flux, wave_water)
            
            
        if np.nanmax(flux_airmass) - np.nanmin(flux_airmass) < self.feature_depth:
            self.has_airmass_features = False
            
        templates[:, 2] = flux_airmass
            
        return templates
    

    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars, templates, wave_final):
        vel = pars[self.par_names[0]].value
        flux = np.ones(templates[:, 0].size)
        if self.has_water_features:
            flux *= self.build_component(pars, templates, 'water')
        if self.has_airmass_features:
            flux *= self.build_component(pars, templates, 'airmass')
        if vel != 0:
            flux = pcmath.doppler_shift(templates[:, 0], wave_out=wave_final, vel=vel, flux=flux, interp='cspline')
        else:
            flux = pcmath.cspline_interp(templates[:, 0], flux, wave_final)
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
        

####################
#### LSF MODELS ####
####################

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
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, blueprint, sregion, order_num, model_dl):

        # Call super method
        super().__init__(blueprint, sregion, order_num)
        
        # Model wavelength grid step size
        self.model_dl = model_dl

    ##################
    #### BUILDERS ####
    ################## 

    def convolve_flux(self, raw_flux, pars=None, lsf=None, interp=False):
        if lsf is None and pars is None:
            raise ValueError("Cannot construct LSF with no parameters")
        if lsf is None:
            lsf = self.build(pars)
        convolved_flux = pcmath.convolve_flux(None, raw_flux, R=None, width=None, interp=interp, lsf=lsf, croplsf=False)
        #convolved_flux = pcmath._convolve(raw_flux, lsf)

        return convolved_flux
            
class HermiteLSF(LSF):
    """ A Hermite Gaussian LSF model. The model is a sum of Gaussians of constant width and Hermite Polynomial coefficients.

    Attributes:
        hermdeg (int): The degree of the hermite model. Zero corresponds to a pure Gaussian.
    """
    
    name = "hermite_lsf"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, blueprint, sregion, order_num, model_dl):

        # Call super
        super().__init__(blueprint, sregion, order_num, model_dl)

        # The Hermite degree
        self.hermdeg = self.blueprint['hermdeg']
        
        # The x grid
        self.nx = self.blueprint["nx"]
        self.x = np.arange(int(-self.nx / 2), int(self.nx / 2) + 1) * self.model_dl
        
        # Build the lsf to estimate where it is small in flux
        self.n_pad_model = int(np.floor(self.nx / 2))

        # Width
        self.base_par_names = ['_width']

        for k in range(self.hermdeg):
            self.base_par_names.append('_a' + str(k+1))
        self.par_names = [self.name + s for s in self.base_par_names]


    def _init_parameters(self, data):
        pars = BoundedParameters()
        pars[self.par_names[0]] = BoundedParameter(value=self.blueprint['width'][1],
                                                   vary=True,
                                                   lower_bound=self.blueprint['width'][0],
                                                   upper_bound=self.blueprint['width'][2])
        for i in range(self.hermdeg):
            pars[self.par_names[i+1]] = BoundedParameter(value=self.blueprint['ak'][1],
                                                                 vary=True,
                                                                 lower_bound=self.blueprint['ak'][0],
                                                                 upper_bound=self.blueprint['ak'][2])
            
        return pars
    
    

    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars):
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

class ModGaussLSF(LSF):
    """ A Modified Gaussian LSF model.
    """
    
    name = "modgauss_lsf"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, blueprint, sregion, order_num):

        # Call super
        super().__init__(blueprint, sregion, order_num)

        self.base_par_names = ['_width', '_p']
        self.par_names = [self.name + s for s in self.base_par_names]


    def _init_parameters(self, data):
        pars = BoundedParameters()
        pars[self.par_names[0]] = BoundedParameter(value=self.blueprint['width'][1],
                                                       vary=True,
                                                       lower_bound=self.blueprint['width'][0],
                                                       upper_bound=self.blueprint['width'][2])
        pars[self.par_names[1]] = BoundedParameter(value=self.blueprint['p'][1],
                                                       vary=True,
                                                       lower_bound=self.blueprint['p'][0],
                                                       upper_bound=self.blueprint['p'][2])
        
        return pars


    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars):
        width = pars[self.par_names[0]].value
        p = pars[self.par_names[1]].value
        lsf = np.exp(-0.5 * np.abs(self.x / width)**p)
        lsf /= np.nansum(lsf)
        return lsf

class PerfectLSF(LSF):
    """ A container for an LSF known a priori.
    
    Attr:
        default_lsf (np.ndarray): The default LSF model to use.
    """
    
    name = "perfect_lsf"

    def __init__(self, blueprint, sregion, order_num):

        super().__init__(blueprint, sregion, order_num)
        self.base_par_names = []
        self.par_names = []

    def build(self, pars=None):
        return self.default_lsf
    
    def convolve_flux(self, raw_flux, pars=None, lsf=None):
        lsf = build(pars=pars)
        return super().convolve_flux(raw_flux, lsf=lsf)     


####################
#### WLS MODELS ####
####################

class WavelengthSolution(SpectralComponent):
    """ A base class for a wavelength solution (i.e., conversion from pixels to wavelength).

    Attributes:
        pix_bounds (list): The left and right pixel bounds which correspond to wave_bounds.
        nx (int): The total number of data pixels.
        default_wave_grid (np.ndarray): The default wavelength grid to use or start from.
    """
    
    key = "wavelength_solution"
    
    def __init__(self, blueprint, sregion, order_num, wls_estimate):
        
        super().__init__(blueprint, sregion, order_num)
        
        self.wls_estimate = wls_estimate
        self.nx = len(self.wls_estimate)

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
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, blueprint, sregion, order_num, wls_estimate):

        # Call super method
        super().__init__(blueprint, sregion, order_num, wls_estimate)
        
        # The polynomial order
        self.poly_order = self.blueprint['poly_order']
        self.n_poly_pars = self.poly_order + 1
            
        # Parameter names
        self.base_par_names = []
        
        # Polynomial lagrange points
        self.poly_pixel_lagrange_points = np.linspace(self.sregion.pixmin, self.sregion.pixmax, num=self.n_poly_pars).astype(int)
        self.poly_wave_lagrange_zero_points = wls_estimate[self.poly_pixel_lagrange_points]
        
        # Base parameter names
        for i in range(self.n_poly_pars):
            self.base_par_names.append('_poly_lagrange_' + str(i + 1))
                
        # Parameter names
        self.par_names = [self.name + s for s in self.base_par_names]

    def _init_parameters(self, data):
        pars = BoundedParameters()
        # Poly parameters
        for i in range(self.n_poly_pars):
            pars[self.par_names[i]] = BoundedParameter(value=self.blueprint["poly_wave_lagrange"][1],
                                                       vary=True,
                                                       lower_bound=self.blueprint['poly_wave_lagrange'][0],
                                                       upper_bound=self.blueprint['poly_wave_lagrange'][2])
        return pars


    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars):
        
        # The detector grid
        pixel_grid = np.arange(self.nx)
            
        # Offsets for each Lagrange point
        poly_lagrange_pars = np.array([pars[self.par_names[i]].value for i in range(self.n_poly_pars)])
        
        # Get the coefficients
        V = np.vander(self.poly_pixel_lagrange_points, N=self.n_poly_pars)
        Vinv = np.linalg.inv(V)
        coeffs = np.dot(Vinv, self.poly_wave_lagrange_zero_points + poly_lagrange_pars)
    
        # Build full polynomial
        poly_wave = np.polyval(coeffs, pixel_grid)
        
        return poly_wave

class SplineWavelengthSolution(WavelengthSolution):
    """ Class for a full wavelength solution defined through cubic splines.

    Attributes:
        poly_order (int): The polynomial order.
        n_splines (int): The number of wavelength splines.
        quad_pixel_set_points (np.ndarray): The three pixel points to use as set points in the quadratic.
        quad_wave_zero_points (np.ndarray): Estimates of the corresonding zero points of quad_pixel_set_points.
        spline_pixel_set_points (np.ndarray): The location of the spline knots in pixel space.
    """
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    name = "spline_wls"

    def __init__(self, blueprint, sregion, order_num, wls_estimate):

        # Call super method
        super().__init__(blueprint, sregion, order_num, wls_estimate)
        
        # The number of spline knots is n_splines + 1
        self.n_splines = self.blueprint['n_splines']

        # Parameter names
        self.base_par_names = []

        # Set the spline parameter names and knots
        for i in range(self.n_splines + 1):
            self.base_par_names.append('_spline_' + str(i + 1))
                
        self.par_names = [self.name + s for s in self.base_par_names]
        
        self.spline_pixel_lagrange_points = np.linspace(self.sregion.pixmin, self.sregion.pixmax, num=self.n_splines + 1).astype(int)
        self.spline_wave_lagrange_zero_points = wls_estimate[self.spline_pixel_lagrange_points]

    def _init_parameters(self, data):
        pars = BoundedParameters()
        for i in range(self.n_splines + 1):
            pars[self.par_names[i]] = BoundedParameter(value=self.blueprint['spline_lagrange'][1],
                                                       vary=True,
                                                       lower_bound=self.blueprint['spline_lagrange'][0],
                                                       upper_bound=self.blueprint['spline_lagrange'][2])
            
        return pars

    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars):
        
        # The detector grid
        pixel_grid = np.arange(self.nx)

        # Get the spline parameters
        spline_pars = np.array([pars[self.par_names[i]].value for i in range(self.n_splines + 1)], dtype=np.float64)
        
        # Build the spline model
        spline_wave = pcmath.cspline_interp(self.spline_pixel_lagrange_points,
                                            spline_pars + self.spline_wave_lagrange_zero_points,
                                            pixel_grid)
        
        return spline_wave

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
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, blueprint, sregion, order_num, wls_estimate):

        # Call super method
        super().__init__(blueprint, sregion, order_num, wls_estimate)
        
        # The polynomial order
        self.poly_order = blueprint['poly_order']
            
        # Parameter names
        self.base_par_names = []
        
        # Polynomial lagrange points
        self.poly_pixel_lagrange_points = np.linspace(self.sregion.pixmin, self.sregion.pixmax, num=self.n_poly_pars).astype(int)
        self.poly_wave_lagrange_zero_points = wls_estimate[self.poly_pixel_lagrange_points]
        
        for i in range(self.n_poly_pars):
            self.base_par_names.append('_poly_lagrange_' + str(i + 1))
                
        self.par_names = [self.name + s for s in self.base_par_names]

    def _init_parameters(self, data):
        pars = BoundedParameters()
        for i in range(self.poly_order + 1):
            pars[self.par_names[i]] = BoundedParameter(value=self.blueprint['poly_lagrange'][1],
                                                       vary=True,
                                                       lower_bound=self.blueprint['poly_lagrange'][0],
                                                       upper_bound=self.blueprint['poly_lagrange'][2])
        return pars


    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars):
        
        # The detector grid
        pixel_grid = np.arange(self.nx)
            
        # Lagrange points
        poly_lagrange_pars = np.array([pars[self.par_names[i]].value for i in range(self.poly_order + 1)])
        
        # Get the coefficients
        coeffs = pcmath.leg_coeffs(self.poly_pixel_lagrange_points, self.poly_wave_lagrange_zero_points + poly_lagrange_pars)
    
        # Build full polynomial
        wave_sol = np.zeros(self.nx)
        for i in range(self.n_poly_pars):
            wave_sol += coeffs[i] * eval_legendre(i, pixel_grid)
        
        return wave_sol

class PerfectWavelengthSolution(WavelengthSolution):
    
    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars):
        return self.data.apriori_wave_grid
    
    ####################
    #### INITIALIZE ####
    ####################
    
    def initialize(self, p0, data, templates_dict, iter_index=None):
        self.data = data


######################
#### MISC. MODELS ####
######################

#### Fringing ####
class FPCavityFringing(EmpiricalMult):
    """ A basic Fabry-Perot cavity model.
    """
    
    name = "fp_fringing"
    key = "fringing"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, blueprint, sregion, order_num):

        # Super
        super().__init__(blueprint, sregion, order_num)

        self.base_par_names = ['_logd', '_fin']

        self.par_names = [self.name + s for s in self.base_par_names]

    def _init_parameters(self, data):
        
        pars = BoundedParameters()
        
        pars[self.par_names[0]] = BoundedParameter(value=self.blueprint['logd'][1],
                                                       vary=True,
                                                       lower_bound=self.blueprint['logd'][0],
                                                       upper_bound=self.blueprint['logd'][2])
        pars[self.par_names[1]] = BoundedParameter(value=self.blueprint['fin'][1],
                                                       vary=True,
                                                       lower_bound=self.blueprint['fin'][0],
                                                       upper_bound=self.blueprint['fin'][2])
        
        return pars

    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars, wave_final):
        d = np.exp(pars[self.par_names[0]].value)
        fin = pars[self.par_names[1]].value
        theta = (2 * np.pi / wave_final) * d
        fringing = 1 / (1 + fin * np.sin(theta / 2)**2)
        return fringing
