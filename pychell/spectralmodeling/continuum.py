import numpy as np

from optimize import BoundedParameter, BoundedParameters

import pychell.maths as pcmath
from pychell.spectralmodeling import SpectralModelComponent1d


class Continuum(SpectralModelComponent1d):
    """Base class for a continuum model.
    """


class PolyContinuum(Continuum):
    """Blaze transmission model through a polynomial.
    """

    __slots__ = ['par_names', 'deg', 'coeffs', 'wave0']
    
    def __init__(self, deg=2, coeffs={0: [0.95, 1.0, 1.2], 1: [-0.5, 0.1, 0.5], 2: [-0.1, 0.01, 0.1]}):
        """Initiate a polynomial continuum model.

        Args:
            poly_order (int, optional): The order of the polynomial. Defaults to 4.
        """
        
        # Super
        super().__init__()
        
        # The polynomial order
        self.deg = deg
        self.coeffs = coeffs
            
        # Parameter names
        for i in range(self.deg + 1):
            self.par_names.append(f"cont_poly_{i}")

    def get_init_parameters(self, data, templates, sregion):

        self.wave0 = np.nanmean(templates['wave'])
        
        # Parameters
        pars = BoundedParameters()
        
        # Poly parameters
        for i in range(self.deg + 1):
            if i in self.coeffs:
                pars[self.par_names[i]] = BoundedParameter(value=self.coeffs[i][1],
                                                            vary=True,
                                                            lower_bound=self.coeffs[i][0], upper_bound=self.coeffs[i][2])
            else:
                prev = pars[self.par_names[i - 1]]
                pars[self.par_names[i]] = BoundedParameter(value=prev.value/10,
                                                           vary=True,
                                                           lower_bound=prev.lower_bound/10, upper_bound=prev.upper_bound/10)
        
        return pars

    def build(self, pars, wave_final):
        
        # The polynomial coeffs
        poly_pars = np.array([pars[self.par_names[i]].value for i in range(self.deg + 1)])
        
        # Build polynomial (and flip coeffs for numpy)
        poly_cont = np.polyval(poly_pars[::-1], wave_final - self.wave0)
        
        return poly_cont

    def __repr__(self):
        return f"Polynomial continuum: deg = {self.deg}"


class SplineContinuum(Continuum):
    """  Blaze transmission model through a polynomial and/or splines, ideally used after a flat field correction or after remove_continuum but not required.
    """

    __slots__ = ['par_names', 'n_splines', 'spline', 'wave_knots']

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
            self.par_names.append(f"cont_spline_{i+1}")

    def get_init_parameters(self, data, templates, sregion):
        self.wave_knots = np.linspace(sregion.wavemin, sregion.wavemax, num=self.n_splines+1)
        pars = BoundedParameters()
        for ispline in range(self.n_splines + 1):
            pars[self.par_names[ispline]] = BoundedParameter(value=self.spline[1],
                                                             vary=True,
                                                             lower_bound=self.spline[0],
                                                             upper_bound=self.spline[2])
        return pars
    
    
    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars, wave_final):

        # Get the spline parameters
        spline_pars = np.array([pars[self.par_names[i]].value for i in range(self.n_splines + 1)], dtype=np.float64)

        # Build
        spline_cont = pcmath.cspline_interp(self.wave_knots, spline_pars, wave_final)
        
        return spline_cont

    def __repr__(self):
        return f"Cubic spline continuum: n_splines = {self.n_splines}"



def estimate_continuum_wobble1d(wave, flux, mask, deg=4, n_sigma=(0.3,3.0), max_iters=50):
    """Fit the continuum using sigma clipping. This function is nearly identical to Megan Bedell's Wobble code.
    Args:
        x (np.ndarray): The wavelengths.
        y (np.ndarray): The fluxes.
        deg (int): The polynomial order to use
        n_sigma (tuple): The sigma clipping threshold: (low, high)
        max_iters: The maximum number of iterations to do.
    Returns:
        The value of the continuum at the wavelengths in x in log space.
    """
    
    # Copy the wave and flux
    x = np.copy(wave)
    y = np.copy(flux)
    
    # Smooth the flux first
    y = pcmath.median_filter1d(y, 3, preserve_nans=True)
    
    # Create a Vander Matrix to solve
    V = np.vander(x - np.nanmean(x), deg + 1)
    
    # Mask to update
    maskcp = np.copy(mask)

    # Make sure the mask is correct
    bad = np.where(~np.isfinite(x) | ~np.isfinite(y))[0]
    if bad.size > 0:
        maskcp[bad] = 0
    
    # Iteratively solve for continuum
    for i in range(max_iters):

        # Get good indices
        good = np.where(maskcp)[0]
        
        # Solve for continuum
        w = np.linalg.solve(np.dot(V[good].T, V[good]), np.dot(V[good].T, y[good]))
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


def estimate_continuum_splines1d(wave, flux, window=None, n_knots=4):
    good = np.where(np.isfinite(wave) & np.isfinite(flux))[0]
    w, f = wave[good], flux[good]
    f = pcmath.median_filter1d(f, width=5)
    continuum = np.full(len(wave), np.nan)
    if window is None:
        window = (np.max(w) - np.min(w)) / (2 * n_knots)
    continuum[good] = pcmath.cspline_fit_fancy(w, f, window, n_knots, percentile=0.99)
    return continuum

