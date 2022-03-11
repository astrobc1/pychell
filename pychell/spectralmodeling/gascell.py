import numpy as np

from optimize import BoundedParameter, BoundedParameters

import pychell.maths as pcmath
from pychell.spectralmodeling import SpectralModelComponent1d


class GasCell(SpectralModelComponent1d):
    """A base class for a gas cell model.
    """

    __slots__ = ['input_file', 'shift', 'depth']
    
    def __init__(self, input_file, shift=[0,0,0], depth=[1,1,1]):

        # Call super method
        super().__init__()
        
        self.shift = shift
        self.depth = depth
        self.input_file = input_file
        self.par_names = ["shift", "depth"]

    def get_init_parameters(self, data, templates, sregion):
        pars = BoundedParameters()
        pars[self.par_names[0]] = BoundedParameter(value=self.shift[1], vary=True,
                                                   lower_bound=self.shift[0], upper_bound=self.shift[2])
        pars[self.par_names[1]] = BoundedParameter(value=self.depth[1], vary=True,
                                                   lower_bound=self.depth[0], upper_bound=self.depth[2])
        
        return pars
        
    def load_template(self, wave_out):
        print('Loading Gas Cell Template', flush=True)
        template_raw = np.load(self.input_file)
        wave, flux = template_raw['wavelength'], template_raw['flux']
        wi, wf = np.nanmin(wave_out), np.nanmax(wave_out)
        good = np.where((wave > wi) & (wave < wf))[0]
        wave, flux = wave[good], flux[good]
        flux = pcmath.cspline_interp(wave, flux, wave_out)
        flux /= pcmath.weighted_median(flux, percentile=0.999)
        template_raw.close()
        return flux

    def build(self, pars, templates):
        flux = templates["gascell"]
        vel = pars[self.par_names[0]].value
        depth = pars[self.par_names[1]].value
        if depth != 1:
            flux = flux**depth
        if vel != 0:
            flux = pcmath.doppler_shift_flux(templates['wave'], flux, vel)
        return flux

    def __repr__(self):
        return f"Gas Cell: {self.input_file}, depth={self.depth}, shift={self.shift}"