import numpy as np

from optimize import BoundedParameter, BoundedParameters

import pychell.maths as pcmath
import pychell.utils as pcutils
from pychell.spectralmodeling import SpectralModelComponent1d

#####################
#### STAR MODELS ####
#####################

class Star(SpectralModelComponent1d):
    """A base class for a star model.
    """
    pass


class AugmentedStar(Star):
    """ A star model which may be augmented after each iteration according to the augmenter attribute in the SpectralRVProb object.
    """

    __slots__ = ['input_file', 'star_name', 'vel_bounds', 'rv_abs', 'par_names']

    def __init__(self, input_file=None, star_name=None, vel_bounds=[-10_000, 10_000], rv_abs=None):

        # Call super method
        super().__init__()
        
        # Input file
        self.input_file = input_file

        # Star info
        self.star_name = star_name

        # Vel bounds
        self.vel_bounds = vel_bounds

        # Absolute RV
        self.rv_abs = rv_abs

        # Pars
        self.par_names = ['vel_star']

    def get_init_parameters(self, data, templates, sregion):
        pars = BoundedParameters()
        if not self.from_flat:
            if self.rv_abs is None:
                rv_absolute = pcutils.get_stellar_rv(self.star_name)
            else:
                rv_absolute = self.rv_abs
            v = rv_absolute - data.header["bc_vel"]
        else:
            v = -1 * data.header["bc_vel"]
        pars[self.par_names[0]] = BoundedParameter(value=v, vary=not self.from_flat,
                                                   lower_bound=v + self.vel_bounds[0],
                                                   upper_bound=v + self.vel_bounds[1])
    
        return pars
        
    def load_template(self, wave_out):
        if not self.from_flat:
            print("Loading Stellar Template", flush=True)
            template_raw = np.loadtxt(self.input_file, delimiter=',')
            wave, flux = template_raw[:, 0], template_raw[:, 1]
            wi, wf = np.nanmin(wave_out), np.nanmax(wave_out)
            good = np.where((wave > wi) & (wave < wf))[0]
            wave, flux = wave[good], flux[good]
            flux = pcmath.cspline_interp(wave, flux, wave_out)
            flux /= pcmath.weighted_median(flux, percentile=0.999)
            del template_raw
        else:
            flux = np.ones(len(wave_out))
        return flux
        
    def build(self, pars, templates):
        flux = pcmath.doppler_shift_flux(templates['wave'], templates['star'], pars[self.par_names[0]].value)
        return flux


    @property
    def from_flat(self):
        return self.input_file is None
    
    def __repr__(self):
        if self.from_flat:
            return f"Augmented star: (from flat)"
        else:
            return f"Augmented star: {self.input_file}"

