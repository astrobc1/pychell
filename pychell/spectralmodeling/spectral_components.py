# Base Python
import glob

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
    """

    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self):

        # No parameter names, probably overwritten with each instance
        self.base_par_names = []
        self.par_names = []
        
    def _init_parameters(self, data):
        return BoundedParameters()

    ####################
    #### INITIALIZE ####
    ####################

    def initialize(self, spectral_model, iter_index=None):
        pass

    def lock_pars(self, pars):
        for pname in self.par_names:
            pars[pname].vary = False
            
    def vary_pars(self, pars):
        for pname in self.par_names:
            pars[pname].vary = True

    ###############
    #### MISC. ####
    ###############

    def __repr__(self):
        s = f"Spectral model: {self.name}\n"
        s += "Parameters:"
        for pname in self.par_names:
            s += f"{pname}\n"
        return s

class MultModelComponent(SpectralComponent):
    """Base class for a multiplicative (or log-additive) spectral component.
    """
    pass

class EmpiricalMult(MultModelComponent):
    """ Base class for an empirically derived multiplicative spectral component (i.e., based purely on parameters, no base templates involved).
    """
    pass

class TemplateMult(MultModelComponent):
    """ A base class for a template based multiplicative model.
    """
    
    #############################
    #### CONSTRUCTOR HELPERS ####
    #############################
    
    def _init_template(self, *args, **kwargs):
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

class Continuum(EmpiricalMult):
    """Base class for a continuum model.
    """
    
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

class PolyContinuum(Continuum):
    """Blaze transmission model through a polynomial.
    """
    
    name = "polynomial_continuum"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, poly_order=4):
        """Initiate a polynomial continuum model.

        Args:
            poly_order (int, optional): The order of the polynomial. Defaults to 4.
        """
        
        # Super
        super().__init__()
        
        # The polynomial order
        self.poly_order = poly_order
            
        # Parameter names
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

class SplineContinuum(Continuum):
    """  Blaze transmission model through a polynomial and/or splines, ideally used after a flat field correction or after remove_continuum but not required.
    """
    
    name = "spline_continuum"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, n_splines=6, spline=[0.3, 1.0, 1.2]):
        """Initiate a spline continuum model.

        Args:
            n_splines (int, optional): The number of splines. The number of knots (parameters) is equal to n_splines + 1. Defaults to 6.
            spline (list, optional): The lower bound, starting value, and upper bound. Defaults to [0.3, 1.0, 1.2].
        """
        
        # Super
        super().__init__()

        # The number of spline knots is n_splines + 1
        self.n_splines = n_splines
        
        # The range for each spline
        self.spline = spline

        # Set the spline parameter names and knots
        for i in range(self.n_splines+1):
            self.base_par_names.append(f"_spline_{i+1}")

        self.par_names = [self.name + s for s in self.base_par_names]

    def _init_parameters(self, data):
        pars = BoundedParameters()
        for ispline in range(self.n_splines + 1):
            pars[self.par_names[ispline]] = BoundedParameter(value=1.0,
                                                             vary=True,
                                                             lower_bound=0.25,
                                                             upper_bound=1.2)
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
    
    
    ####################
    #### INITIALIZE ####
    ####################
    
    def initialize(self, spectral_model, iter_index=None):
        self.spline_wave_set_points = np.linspace(spectral_model.sregion.wavemin, spectral_model.sregion.wavemax, num=self.n_splines + 1)


#########################
#### GAS CELL MODELS ####
#########################

class GasCell(TemplateMult):
    """A base class for a gas cell model.
    """
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, input_file):

        # Call super method
        super().__init__()
        
        self.input_file = input_file
        
    def _init_template(self, data, sregion, model_dl):
        print('Loading Gas Cell Template', flush=True)
        pad = 5
        template = np.load(self.input_file)
        wave, flux = template['wave'], template['flux']
        good = np.where((wave > sregion.wavemin - pad) & (wave < sregion.wavemax + pad))[0]
        wave, flux = wave[good], flux[good]
        flux /= pcmath.weighted_median(flux, percentile=0.999)
        template = np.array([wave, flux]).T
        return template

class DynamicGasCell(GasCell):
    """A dynamic gas cell model allowing for a depth and shift.
    """
    
    name = "dynamic_gas_cell"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, input_file=None, shift=[0, 0, 0], depth=[1, 1, 1]):
        """Initiate a dynamic gas cell model.

        Args:
            input_file ([type], optional): The full path to the gas cell template. Defaults to None.
            shift (list, optional): The lower bound, starting value, and upper bound for the gas cell shift in Angstroms. Defaults to [0, 0, 0].
            depth (list, optional): The lower bound, starting value, and upper bound for the gas cell depth. Defaults to [1, 1, 1].
        """

        # Call super method
        super().__init__(input_file=input_file)
        
        self.shift = shift
        self.depth = depth

        self.base_par_names += ['_shift', '_depth']
        self.par_names = [self.name + s for s in self.base_par_names]

    def _init_parameters(self, data):
        pars = BoundedParameters()
        pars[self.par_names[0]] = BoundedParameter(value=self.shift[1], vary=True,
                                                   lower_bound=self.shift[0], upper_bound=self.shift[2])
        pars[self.par_names[1]] = BoundedParameter(value=self.depth[1], vary=True,
                                                   lower_bound=self.depth[0], upper_bound=self.depth[2])
        
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
    """A perfect gas cell model (no modifications).
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
    """A base class for a star model.
    """
    pass

class AugmentedStar(Star):
    """ A star model which may be augmented after each iteration according to the augmenter attribute in the SpectralRVProb object.
    """
    
    name = "augmented_star"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, input_file=None):

        # Call super method
        super().__init__()
        
        # Input file
        self.input_file = input_file
        
        # Whether or not the star is from a synthetic source
        self.from_flat = True if self.input_file is None else False

        # Pars
        self.base_par_names += ['_vel']
        
        # Update parameter names
        self.par_names = [self.name + s for s in self.base_par_names]

    def _init_parameters(self, data):
        pars = BoundedParameters()
        pars[self.par_names[0]] = BoundedParameter(value=100, vary=True,
                                                   lower_bound=-3E5, upper_bound=3E5)
    
        return pars
        
    def _init_template(self, data, sregion, model_dl):
        pad = 15
        wave_uniform = np.arange(sregion.wavemin - pad, sregion.wavemax + pad, model_dl)
        if not self.from_flat:
            print("Loading Stellar Template", flush=True)
            template_raw = np.loadtxt(self.input_file, delimiter=',')
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
    
    def initialize(self, spectral_model, iter_index=None):
        if iter_index == 0 and self.from_flat:
            spectral_model.p0[self.par_names[0]].vary = False
        elif iter_index == 1 and self.from_flat:
            spectral_model.p0[self.par_names[0]].vary = True
            spectral_model.p0[self.par_names[0]].value = -1 * data.bc_vel


#########################
#### TELLURIC MODELS ####
#########################

class Tellurics(TemplateMult):
    """A base class for tellurics.
    """
    pass

class TelluricsTAPAS(Tellurics):
    """A telluric model based on templates obtained from TAPAS which are specific to a certain observatory (or generate site such as Maunakea). These templates should be pre-fetched from TAPAS and specific to the site. CH4, N20, CO2, O2, and O3 utilize a common depth parameter. H2O utilizes a unique depth. All species utilize a common Doppler shift.
    """
    
    name = "tapas_tellurics"
    species = ['water', 'methane', 'carbon_dioxide', 'nitrous_oxide', 'oxygen', 'ozone']
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, input_path, location_tag, feature_depth=0.02, vel=[-300, 50, 300], water_depth=[0.05, 1.1, 5.0], airmass_depth=[0.8, 1.1, 3.0]):
        """Initiate a TAPAS telluric model.

        Args:
            input_path (str): The full path to the directory containing the six speciesn files.
            location_tag (str): The location tag present in the filename for each species.
            feature_depth (float, optional): If a set of templates (water, everything else) has a dynamic range of less than feature_depth, that set is ignored. Defaults to 0.02 (2 percent).
            vel (list, optional): The lower bound, starting value, and upper bound for the telluric shift in m/s. Defaults to [-300, 50, 300].
            water_depth (list, optional): The lower bound, starting value, and upper bound for the water depth. Defaults to [0.05, 1.1, 5.0].
            airmass_depth (list, optional): The lower bound, starting value, and upper bound for the species which correlate well with airmass (everything but water). Defaults to [0.8, 1.1, 3.0].
        """
        super().__init__()
        self.input_path = input_path
        self.location_tag = location_tag
        self.vel = vel
        self.water_depth = water_depth
        self.airmass_depth = airmass_depth
        self.base_par_names += ['_vel', '_water_depth', '_airmass_depth']
        self.has_water_features, self.has_airmass_features = True, True
        self.feature_depth = feature_depth
        self.par_names = [self.name + s for s in self.base_par_names]
        
        # Input files
        self.species_input_files = {species: f"{self.input_path}telluric_{species}_tapas_{self.location_tag}.npz" for species in self.species}

    def _init_parameters(self, data):
        
        pars = BoundedParameters()
        
        # Velocity
        pars[self.par_names[0]] = BoundedParameter(value=self.vel[1],
                                                   vary=(self.has_water_features or self.has_airmass_features),
                                                   lower_bound=self.vel[0], upper_bound=self.vel[2])
        
        # Water Depth
        pars[self.par_names[1]] = BoundedParameter(value=self.water_depth[1],
                                                   vary=self.has_water_features,
                                                   lower_bound=self.water_depth[0], upper_bound=self.water_depth[2])
        
        # Remaining Components
        pars[self.par_names[2]] = BoundedParameter(value=self.airmass_depth[1],
                                                   vary=self.has_airmass_features,
                                                   lower_bound=self.airmass_depth[0], upper_bound=self.airmass_depth[2])
        
        return pars

    def _init_template(self, data, sregion, model_dl):
        print('Loading Telluric Templates', flush=True)
        # Pad
        pad = 5
        
        # Water
        water = np.load(self.species_input_files['water'])
        wave, flux = water['wave'], water['flux']
        good = np.where((wave > sregion.wavemin - pad) & (wave < sregion.wavemax + pad))[0]
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
            tell = np.load(self.species_input_files[species])
            wave, _flux = tell['wave'], tell['flux']
            good = np.where((wave > sregion.wavemin - pad) & (wave < sregion.wavemax + pad))[0]
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
        
# class TelluricsDev(Tellurics):
#     pass

#    def __init__(self, input_path, feature_depth=0.01, ):
#        pass

#    def build(self, pars):
#        pass

#    def initialize(self, pars):
#        pass

####################
#### LSF MODELS ####
####################

class LSF(SpectralComponent):
    """ A base class for an LSF (line spread function) model.
    """
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self):

        # Call super method
        super().__init__()

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
    """A Hermite Gaussian LSF model. The model is a sum of Gaussians of constant width with Hermite Polynomial coefficients to enforce orthogonality. See Arfken et al. for more details.
    """
    
    name = "hermite_lsf"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, hermdeg=0, width=None, hermcoeff=[-0.1, 0.01, 0.1]):
        """Initate a Hermite LSF model.

        Args:
            hermdeg (int, optional): The degree of the Hermite polynomials. Defaults to 0, which is identical to a standard Gaussian.
            width (float, optional): The lower bound, starting value, and upper bound of the LSF width in Angstroms. Defaults to None.
            hermcoeff (list, optional): The lower bound, starting value, and upper bound for each Hermite polynomial coefficient. Defaults to [-0.1, 0.01, 0.1].
        """

        # Call super
        super().__init__()

        # The Hermite degree
        self.hermdeg = hermdeg
        self.width = width
        self.hermcoeff = hermcoeff

        # Width
        self.base_par_names = ['_width']

        for k in range(self.hermdeg):
            self.base_par_names.append('_a' + str(k+1))
        self.par_names = [self.name + s for s in self.base_par_names]

    def _init_parameters(self, data):
        pars = BoundedParameters()
        pars[self.par_names[0]] = BoundedParameter(value=self.width[1], vary=True,
                                                   lower_bound=self.width[0], upper_bound=self.width[2])
        for i in range(self.hermdeg):
            pars[self.par_names[i+1]] = BoundedParameter(value=self.hermcoeff[1], vary=True,
                                                         lower_bound=self.hermcoeff[0], upper_bound=self.hermcoeff[2])
            
        return pars
    
    
    ####################
    #### INITIALIZE ####
    ####################
    
    def initialize(self, spectral_model, iter_index=None):
        nx = spectral_model.data.flux.size
        self.x = np.arange(int(-nx / 2), int(nx / 2) + 1)
        lsf_init = self.build(spectral_model.p0)
        lsf_init /= np.nanmax(lsf_init) # norm to max
        good = np.where(lsf_init > 1E-10)[0]
        x_min, x_max = good.min(), good.max()
        nx = int(2 * np.max([np.abs(x_min), x_max])) # or +/- 1
        if nx % 2 == 0:
            nx += 1
        self.x = np.arange(int(-nx / 2), int(nx / 2) + 1) * spectral_model.model_dl
        self.n_pad_model = int(np.floor(self.x.size / 2))
    
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

class PerfectLSF(LSF):
    """A model for a perect LSF (known a priori).
    """
    
    name = "perfect_lsf"

    def build(self, pars=None):
        return self.default_lsf
    
    def convolve_flux(self, raw_flux, pars=None, lsf=None):
        lsf = build(pars=pars)
        return super().convolve_flux(raw_flux, lsf=lsf)     


####################
#### WLS MODELS ####
####################

class WavelengthSolution(SpectralComponent):
    """A base class for a wavelength solution (conversion from pixels to wavelength).
    """
    pass

class PolyWavelengthSolution(WavelengthSolution):
    """A polynomial wavelength solution model. Instead of optimizing coefficients, the model utilizes set points which are evenly spaced across the spectral range in pixel space.
    """
    
    name = "poly_wls"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, poly_order=2, set_point=[-0.5, 0.05, 0.5]):
        """Initiate a polynomial wavelength solution model.

        Args:
            poly_order (int, optional): The order of the polynomial. Defaults to 2.
            set_point (list, optional): The window for each point in Angstroms.
        """

        # Call super method
        super().__init__()
        
        # The polynomial order
        self.poly_order = poly_order
        self.set_point = set_point
        
        # Base parameter names
        for i in range(self.poly_order + 1):
            self.base_par_names.append('_poly_lagrange_' + str(i + 1))
                
        # Parameter names
        self.par_names = [self.name + s for s in self.base_par_names]

    def _init_parameters(self, data):
        pars = BoundedParameters()
        # Poly parameters
        for i in range(self.poly_order + 1):
            pars[self.par_names[i]] = BoundedParameter(value=self.set_point[1],
                                                       vary=True,
                                                       lower_bound=self.set_point[0],
                                                       upper_bound=self.set_point[1])
        return pars


    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars):
        
        # The detector grid
        pixel_grid = np.arange(self.nx)
        
        # The number of parameters
        n_pars = self.poly_order + 1
            
        # Offsets for each Lagrange point
        poly_lagrange_pars = np.array([pars[self.par_names[i]].value for i in range(n_pars)])
        
        # Get the coefficients
        V = np.vander(self.poly_pixel_lagrange_points, N=n_pars)
        Vinv = np.linalg.inv(V)
        coeffs = np.dot(Vinv, self.poly_wave_lagrange_zero_points + poly_lagrange_pars)
    
        # Build full polynomial
        poly_wave = np.polyval(coeffs, pixel_grid)
        
        return poly_wave
    
    ####################
    #### INITIALIZE ####
    ####################
    
    def initialize(self, spectral_model, iter_index=None):
        self.poly_pixel_lagrange_points = np.linspace(spectral_model.sregion.pixmin,
                                                      spectral_model.sregion.pixmax,
                                                      num=self.poly_order + 1).astype(int)
        wls_estimate = spectral_model.data.parser.estimate_wavelength_solution(spectral_model.data)
        self.nx = len(wls_estimate)
        self.poly_wave_lagrange_zero_points = wls_estimate[self.poly_pixel_lagrange_points]

class SplineWavelengthSolution(WavelengthSolution):
    """A cubic spline wavelength solution model.
    """
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    name = "spline_wls"

    def __init__(self, n_splines=6, spline=[-0.5, 0.1, 0.5]):
        """Initiate a spline wavelength solution model.

        Args:
            n_splines (int, optional): The number of splines to use where the number of knots = n_splines + 1. Defaults to 6.
            spline (list, optional): The lower bound, starting value, and upper bound for each spline in Angstroms, and relative to the initial wavelength solution provided from the parser object. Defaults to [-0.5, 0.1, 0.5].
        """

        # Call super method
        super().__init__()
        
        # The number of spline knots is n_splines + 1
        self.n_splines = n_splines
        self.spline = spline

        # Set the spline parameter names and knots
        for i in range(self.n_splines + 1):
            self.base_par_names.append(f"_spline_{i + 1}")
                
        self.par_names = [self.name + s for s in self.base_par_names]

    def _init_parameters(self, data):
        pars = BoundedParameters()
        for i in range(self.n_splines + 1):
            pars[self.par_names[i]] = BoundedParameter(value=self.spline[1], vary=True,
                                                       lower_bound=self.spline[0], upper_bound=self.spline[2])
            
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
    
    ####################
    #### INITIALIZE ####
    ####################
    
    def initialize(self, spectral_model, iter_index=None):
        self.spline_pixel_lagrange_points = np.linspace(spectral_model.sregion.pixmin,
                                                        spectral_model.sregion.pixmax,
                                                        num=self.n_splines + 1).astype(int)
        wls_estimate = spectral_model.data.parser.estimate_wavelength_solution(spectral_model.data)
        self.nx = len(wls_estimate)
        self.spline_wave_lagrange_zero_points = wls_estimate[self.spline_pixel_lagrange_points]

class LegPolyWavelengthSolution(WavelengthSolution):
    """A Legendre polynomial wavelength solution model.
    """
    
    name = "legpoly_wls"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, poly_order=4):
        """Initiate a Legendre polynomial model.

        Args:
            poly_order (int, optional): The order of the Legendre polynomial. Defaults to 2.
        """

        # Call super method
        super().__init__()
        
        # The polynomial order
        self.poly_order = poly_order
            
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
    """A model for a perfect wavelenth solution model (known a priori).
    """
    
    name = "apriori_wls"
    
    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars):
        return self.data.apriori_wave_grid
    
    ####################
    #### INITIALIZE ####
    ####################
    
    def initialize(self, spectral_model, iter_index=None):
        self.data = spectral_model.data


######################
#### MISC. MODELS ####
######################

#### Fringing ####
class FPCavityFringing(EmpiricalMult):
    """A basic Fabry-Perot cavity model for fringing in spectrographs like iSHELL and NIRSPEC.
    """
    
    name = "fp_fringing"
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self):

        # Super
        super().__init__()

        self.base_par_names += ['_logd', '_fin']

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
