import numpy as np

from optimize import BoundedParameter, BoundedParameters

import pychell.maths as pcmath
from pychell.spectralmodeling import SpectralModelComponent1d

class LSF(SpectralModelComponent1d):
    """ A base class for an LSF (line spread function) model.
    """

    def convolve(self, raw_flux, pars=None, lsf=None, interp=False):
        if lsf is None and pars is None:
            raise ValueError("Cannot construct LSF with no parameters")
        if lsf is None:
            lsf = self.build(pars)
        flux = pcmath.convolve1d(raw_flux, lsf) # reforward this method to ignore the options of pcmath.convolve_flux
        return flux
            
class HermiteLSF(LSF):
    """A Hermite Gaussian LSF model. The model is a sum of Gaussians of constant width with Hermite Polynomial coefficients to enforce orthogonality. See Arfken et al. for more details.
    """

    __slots__ = ['deg', 'sigma', 'coeff', 'wave_rel']

    def __init__(self, deg=0, sigma=None, coeff=[-0.1, 0.01, 0.1]):
        """Initate a Hermite LSF model.

        Args:
            hermdeg (int, optional): The degree of the Hermite polynomials. Defaults to 0, which is identical to a standard Gaussian.
            width (float, optional): The lower bound, starting value, and upper bound of the LSF width in Angstroms. Defaults to None.
            hermcoeff (list, optional): The lower bound, starting value, and upper bound for each Hermite polynomial coefficient. Defaults to [-0.1, 0.01, 0.1].
        """

        # Call super
        super().__init__()

        # The Hermite degree
        self.deg = deg
        self.sigma = sigma
        self.coeff = coeff

        # sigma
        self.par_names = ['a0']

        for k in range(self.deg):
            self.par_names.append('a' + str(k+1))

    def get_init_parameters(self, data, templates, sregion):
        dl = templates["wave"][2] - templates["wave"][1]
        self.wave_rel = self.get_wave_grid(sregion, dl)
        pars = BoundedParameters()
        pars[self.par_names[0]] = BoundedParameter(value=self.sigma[1], vary=True,
                                                   lower_bound=self.sigma[0], upper_bound=self.sigma[2])
        for i in range(self.deg):
            pars[self.par_names[i+1]] = BoundedParameter(value=self.coeff[1], vary=True,
                                                         lower_bound=self.coeff[0], upper_bound=self.coeff[2])
            
        return pars

    def get_wave_grid(self, sregion, dl):
        nx = int(14 * self.sigma[2] / dl)
        if nx % 2 == 0:
            nx += 1
        wave_rel = np.arange(int(-nx / 2), int(nx / 2) + 1) * dl
        return wave_rel

    def build(self, pars, wave_rel=None):
        if wave_rel is None:
            wave_rel = self.wave_rel
        sigma = pars[self.par_names[0]].value
        herm = pcmath.hermfun(self.wave_rel / sigma, self.deg)
        if self.deg == 0:  # just a Gaussian
            return herm / np.nansum(herm)
        else:
            kernel = herm[:, 0]
        for i in range(self.deg):
            kernel += pars[self.par_names[i+1]].value * herm[:, i+1]
        return kernel / np.nansum(kernel)

    def __repr__(self):
        return f"Hermite Gaussian LSF: deg = {self.deg}"