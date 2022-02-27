import os

import numpy as np

from optimize import BoundedParameter, BoundedParameters

import pychell.maths as pcmath
from pychell.spectralmodeling import SpectralModelComponent1d

#########################
#### TELLURIC MODELS ####
#########################

class Tellurics(SpectralModelComponent1d):
    """A base class for tellurics.
    """
    pass

class TelluricsTAPAS(Tellurics):
    """A telluric model based on templates obtained from TAPAS which are specific to a certain observatory (or generate site such as Maunakea). These templates should be pre-fetched from TAPAS and specific to the site. CH4, N20, CO2, O2, and O3 utilize a common depth parameter. H2O utilizes a unique depth. All species utilize a common Doppler shift.
    """

    species = ['water', 'methane', 'carbon_dioxide', 'nitrous_oxide', 'oxygen', 'ozone']

    __slots__ = ["input_file", "feature_depth", "vel", "water_depth", "airmass_depth", "mask", "has_water_features", "has_airmass_features"]
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, input_file, feature_depth=0.02, vel=[-300, 50, 300], water_depth=[0.05, 1.1, 5.0], airmass_depth=[0.8, 1.1, 3.0], mask=False):
        """Initiate a TAPAS telluric model.

        Args:
            input_file (str): The full path to the directory containing the six speciesn files.
            feature_depth (float, optional): If a set of templates (water, everything else) has a dynamic range of less than feature_depth, that set is ignored. Defaults to 0.02 (2 percent).
            vel (list, optional): The lower bound, starting value, and upper bound for the telluric shift in m/s. Defaults to [-300, 50, 300].
            water_depth (list, optional): The lower bound, starting value, and upper bound for the water depth. Defaults to [0.05, 1.1, 5.0].
            airmass_depth (list, optional): The lower bound, starting value, and upper bound for the species which correlate well with airmass (everything but water). Defaults to [0.8, 1.1, 3.0].
            mask (bool, optional): Whether or not to mask telluric features instead of modeling them. If True, regions with flux less than 1 - feature_depth are masked when modeling.
        """
        super().__init__()
        self.input_file = input_file
        self.vel = vel
        self.water_depth = water_depth
        self.airmass_depth = airmass_depth
        self.par_names = ['velt', 'water_depth', 'airmass_depth']
        self.has_water_features, self.has_airmass_features = True, True
        self.feature_depth = feature_depth
        self.mask = mask

    def get_init_parameters(self, data, templates, sregion):
        
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

    def load_template(self, wave_out):

        print('Loading Telluric Template', flush=True)

        templates = np.zeros(shape=(len(wave_out), 2), dtype=float)
        
        # Water
        template_raw = np.load(self.input_file)
        wave = template_raw["wavelength"]
        flux_water = template_raw["water"]
        wi, wf = np.nanmin(wave_out), np.nanmax(wave_out)
        good = np.where((wave > wi) & (wave < wf))[0]
        wave, flux_water = wave[good], flux_water[good]
        flux_water_temp = np.copy(flux_water)
        flux_water_temp /= np.nanmax(flux_water_temp)
        if np.nanmax(flux_water_temp) - np.nanmin(flux_water_temp) < self.feature_depth:
            self.has_water_features = False
        flux_water = pcmath.cspline_interp(wave, flux_water, wave_out)
        flux_water /= pcmath.weighted_median(flux_water, percentile=0.999)
        templates[:, 0] = flux_water
        
        # Remaining, do in a loop...
        flux_airmass = np.ones(wave_out.size)
        for s in self.species:
            if s != 'water':
                flux_airmass *= pcmath.cspline_interp(wave, template_raw[s][good], wave_out)

        if np.nanmax(flux_airmass) - np.nanmin(flux_airmass) < self.feature_depth:
            self.has_airmass_features = False
            
        flux_airmass /= pcmath.weighted_median(flux_airmass, percentile=0.999)
        templates[:, 1] = flux_airmass
            
        return templates
    

    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars, templates):
        vel = pars[self.par_names[0]].value
        depth_water = pars[self.par_names[1]].value
        depth_airmass = pars[self.par_names[2]].value
        flux = templates['tellurics'][:, 0]**depth_water * templates['tellurics'][:, 1]**depth_airmass
        if vel != 0:
            flux = pcmath.doppler_shift_flux(templates['wave'], flux, vel)
        return flux


    ##############
    #### MASK ####
    ##############

    def mask_tellurics(self, pars, templates, wave_out=None):
        wave = templates[:, 0]
        if wave_out is None:
            wave_out = wave
        mask = np.ones(len(wave_out))
        flux = self.build(pars, templates, wave_out)
        bad = np.where(flux < 1 - self.feature_depth)[0]
        mask[bad] = 0
        return mask


    def __repr__(self):
        return f"Static TAPAS tellurics: {os.path.basename(self.input_file)}"

