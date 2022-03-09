import numpy as np

from optimize import BoundedParameter, BoundedParameters

import pychell.maths as pcmath
from pychell.spectralmodeling import SpectralModelComponent1d

class WavelengthSolution(SpectralModelComponent1d):
    """A base class for a wavelength solution (conversion from pixels to wavelength).
    """
    pass


class PolyWls(WavelengthSolution):
    """A polynomial wavelength solution model. Instead of optimizing coefficients, the model utilizes set points which are evenly spaced across the spectral range in pixel space.
    """

    __slots__ = ["par_names", "deg", "bounds", "x", "pixel_knots"]

    def __init__(self, deg=2, bounds=[-0.5, 0.5]):
        """Initiate a polynomial wavelength solution model.

        Args:
            deg (int, optional): The order of the polynomial. Defaults to 2.
            bounds (list, optional): The relative bounds for each point in nm. For example, [-0.01, 0.01] would bound each pixel knot by 0.01 nm, which is usually appropriate for most stable echelle spectrographs.
        """

        # Call super method
        super().__init__()
        
        # The polynomial order
        self.deg = deg
        self.bounds = bounds
        
        # Base parameter names
        for i in range(self.deg + 1):
            self.par_names.append('wls_poly_knot_' + str(i + 1))

    def get_init_parameters(self, data, templates, sregion):
        pars = BoundedParameters()
        self.pixel_knots = np.linspace(sregion.pixmin + 1, sregion.pixmax - 1, num=self.deg+1).astype(int)
        wls_estimate = data.spec_module.estimate_wls(data, sregion)
        self.x = np.arange(sregion.pixmin, sregion.pixmax + 1)
        for i in range(self.deg + 1):
            v = wls_estimate[self.pixel_knots[i] - sregion.pixmin]
            pars[self.par_names[i]] = BoundedParameter(value=v, vary=True,
                                                       lower_bound=v + self.bounds[0],
                                                       upper_bound=v + self.bounds[1])
        return pars


    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars, data):
        V = np.vander(self.pixel_knots, N=self.deg + 1)
        Vinv = np.linalg.inv(V)
        coeffs = np.dot(Vinv, np.array([pars[pname].value for pname in self.par_names]))
        wls = np.polyval(coeffs, self.x)
        return wls

    def __repr__(self):
        return f"Polynomial wavelength solution: poly_order = {self.deg}"


class SplineWls(WavelengthSolution):
    """A cubic spline wavelength solution model.
    """

    __slots__ = ["n_splines", "bounds", "pixel_knots", "x"]

    def __init__(self, n_splines=6, bounds=[-0.5, 0.5]):
        """Initiate a spline wavelength solution model.

        Args:
            n_splines (int, optional): The number of splines to use where the number of knots = n_splines + 1. Defaults to 6.
            spline (list, optional): The lower bound, starting value, and upper bound for each spline in Angstroms, and relative to the initial wavelength solution provided from the parser object. Defaults to [-0.5, 0.1, 0.5].
        """

        # Call super method
        super().__init__()
        
        # The number of spline knots is n_splines + 1
        self.n_splines = n_splines
        self.bounds = bounds

        # Set the spline parameter names and knots
        for i in range(self.n_splines + 1):
            self.par_names.append(f"wls_spline_{i + 1}")

    def get_init_parameters(self, data, templates, sregion):
        pars = BoundedParameters()
        self.pixel_knots = np.linspace(sregion.pixmin + 1, sregion.pixmax - 1, num=self.deg+1).astype(int)
        wls_estimate = data.spec_module.estimate_wls(data, sregion)
        self.x = np.arange(sregion.pixmin, sregion.pixmax + 1)
        for i in range(self.n_splines + 1):
            v = wls_estimate[self.pixel_knots[i] - sregion.pixmin]
            pars[self.par_names[i]] = BoundedParameter(value=v, vary=True,
                                                       lower_bound=v + self.bounds[0],
                                                       upper_bound=v + self.bounds[1])
            
        return pars


    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars, data):

        # Get the spline parameters
        splines = np.array([pars[self.par_names[i]].value for i in range(self.n_splines + 1)], dtype=float)
        
        # Build the spline model
        wls = pcmath.cspline_interp(self.pixel_knots, splines, self.x)

        return wls
    
    def __repr__(self):
        return f"Cubic spline wavelength solution: n_splines = {self.n_splines}"


class APrioriWls(WavelengthSolution):
    """A model for a predetermined wavelenth solution model.
    """

    def build(self, pars, data):
        return data.wave

    def __repr__(self):
        return "a priori wavelength solution"
