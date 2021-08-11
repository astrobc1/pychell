
# Maths
import numpy as np

# LLVM
import numba

# optimize deps
from optimize.models import DeterministicModel, NoiseBasedModel, GPBasedModel

# Pychell deps
import pychell.orbits.planetmaths as planetmath

############################
#### KEPLERIAN RV MODEL ####
############################

class KeplerianRVModel(DeterministicModel):
    
    #####################
    #### CONSTRUCTOR ####
    #####################
    
    def __init__(self, data=None, planets_dict=None):
        super().__init__(data=data ,name="Keplerian Model")
        self.planets_dict = {} if planets_dict is None else planets_dict
        
    ##################
    #### BUILDERS ####
    ##################
    
    def builder(self, pars, t=None):
        if t is None:
            t = self.data.t
        return self.build_planets(pars, t)
         
    def build_planet(self, pars, t, planet_index):
        """Builds a keplerian model for a single planet.

        Args:
            pars (Parameters): The parameters.
            t (np.ndarray): The times.
            planet_dict (dict): The planet dictionary.

        Returns:
            np.ndarray: The RV model for this planet.
        """
        
        # Get the planet parameters on the standard basis.
        planet_pars = self.planets_dict[planet_index]["basis"].to_standard(pars)
        
        # Build and return planet signal
        vels = self.planet_signal(t, *planet_pars)
        
        # Return vels
        return vels
        
    def build_planets(self, pars, t):
        """Builds a model including all planets.

        Args:
            t (np.ndarray): The times.
            pars (Parameters): The parameters.

        Returns:
            np.ndarray: The RV model for this planet
        """
        model_arr = np.zeros_like(t)
        for planet_index in self.planets_dict:
            model_arr += self.build_planet(pars, t, planet_index)
        return model_arr
    
    @staticmethod
    @numba.njit
    def planet_signal(t, per, tp, ecc, w, k):
        """Computes the RV signal of one planet for a given time vector.

        Args:
            t (np.ndarray): The times in units of per.
            k (float): The RV semi-amplitude.
            per (float): The period of the orbit in units of t.
            tc (float): The time of conjunction.
            ecc (float): The eccentricity of the bounded orbit.
            w (float): The angle of periastron
            tp (float): The time of perisatron

        Returns:
            np.ndarray: The rv signal for this planet.
        """
        return planetmath.planet_signal(t, per, tp, ecc, w, k)

    @staticmethod
    def _disable_planet_pars(pars, planets_dict, planet_index):
        """Disables (sets vary=False) inplace for the planet parameters corresponding to planet_index.

        Args:
            pars (Parameters): The parameters.
            planets_dict (dict): The planets dict.
            planet_index (int): The index to disable.
        """
        for par in pars.values():
            for planet_par_name in planets_dict[planet_index]["basis"].names:
                if par.name == planet_par_name + str(planet_index):
                    pars[par.name].vary = False


    ###############
    #### MISC. ####
    ###############
        
    @property
    def n_planets(self):
        return len(self.planets_dict)
    

########################
#### TREND RV MODEL ####
########################

class RVTrend(DeterministicModel):
    
    #####################
    #### CONSTRUCTOR ####
    #####################
    
    def __init__(self, poly_order, data=None, time_zero=None):
        super().__init__(data=data, name="RV Trend Model")
        self.poly_order = poly_order
        self.time_zero = np.nanmedian(data.t) if time_zero is None else time_zero
    
    
    ##################
    #### BUILDERS ####
    ##################
    
    def build_trend_zero(self, pars, t, instname=None):
        
        # Zeros
        trend_zero = np.zeros_like(t)
        
        if self.poly_order is not None:
            # Per-instrument zero points
            if instname is not None:
                assert len(t) == len(self.data[instname].t)
                pname = f"gamma_{instname}"
                trend_zero[:] = pars[pname].value
            else:
                for instname in self.data:
                    pname = f"gamma_{instname}"
                    inds = self.data.indices[instname]
                    trend_zero[inds] = pars[pname].value

        return trend_zero

    def build_global_trend(self, pars, t):
        
        # Init trend
        trend = np.zeros_like(t)
                
        # Build trend
        if self.poly_order > 0:
            for i in range(1, self.poly_order + 1):
                trend += pars[f"gamma_{''.join(str(x) for x in (['d']*i))}ot"].value * (t - self.time_zero)**i
        return trend


    ###############
    #### MISC. ####
    ###############

    def __repr__(self):
        return 'An RV Trend Model'


###################################
#### STANDARD RV MODEL (NO GP) ####
###################################

class CompositeRVModel(NoiseBasedModel):
    
    #####################
    #### CONSTRUCTOR ####
    #####################
    
    def __init__(self, data=None, planets_dict=None, noise_process=None, poly_order=0, time_zero=None, name="Composite RV Model"):
        super().__init__(data=data, name=name)
        self.kep_model = KeplerianRVModel(data=data, planets_dict=planets_dict)
        self.trend_model = RVTrend(data=data, poly_order=poly_order, time_zero=time_zero)
        self.noise_process = noise_process

        
    ##################
    #### BUILDERS ####
    ##################
        
    def builder(self, pars, t=None, include_kep=True, include_trend=True):
        if t is None:
            t = self.data.t
        model_arr = np.zeros_like(t)
        if include_kep:
            model_arr += self.kep_model.builder(pars, t)
        if include_trend:
            model_arr += self.trend_model.build_global_trend(pars, t)
        return model_arr
    
    def build_planet(self, *args, **kwargs):
        return self.kep_model.build_planet(*args, **kwargs)
    
    #####################
    #### DATA ERRORS ####
    #####################
    
    def compute_data_errors(self, pars):
        errors = self.noise_process.compute_data_errors(pars)
        return errors
    
    ###################
    #### RESIDUALS ####
    ###################
    
    def compute_raw_residuals(self, pars):
        
        # Time array
        t = self.data.t
        
        # The raw data rvs
        data_arr = self.data.get_trainable()
        
        # Remove the offsets from the data
        data_arr -= self.trend_model.build_trend_zero(pars, t)
        
        # Build the Keplerian model + trend
        model_arr = self.builder(pars, t)
        
        # Residuals
        residuals = data_arr - model_arr
        
        return residuals
        
    def compute_residuals(self, pars):
        return self.compute_raw_residuals(pars)
        
    
    ####################
    #### INITIALIZE ####
    ####################
    
    def initialize(self, p0):
        self.p0 = p0
        self.kep_model.initialize(self.p0)
        self.trend_model.initialize(self.p0)
        self.noise_process.initialize(self.p0)
    
    ###############
    #### MISC. ####
    ###############
    
    def __repr__(self):
        return "Composite RV Model"
    
    @property
    def planets_dict(self):
        return self.kep_model.planets_dict
    
    @property
    def n_planets(self):
        return len(self.planets_dict)
    

###########################
#### GP BASED RV MODEL ####
###########################

class CompositeGPRVModel(CompositeRVModel, GPBasedModel):
    
    #####################
    #### CONSTRUCTOR ####
    #####################
    
    def __init__(self, data=None, planets_dict=None, noise_process=None, poly_order=0, time_zero=None, name="rvs"):
        super().__init__(data=data, name=name)
        self.kep_model = KeplerianRVModel(data=data, planets_dict=planets_dict)
        self.trend_model = RVTrend(data=data, poly_order=poly_order, time_zero=time_zero)
        self.noise_process = noise_process
    
    def compute_data_errors(self, pars, include_corr_error=False):
        if include_corr_error:
            linpred = self.compute_raw_residuals(pars)
        else:
            linpred = None
        errors = self.noise_process.compute_data_errors(pars, include_corr_error=include_corr_error, linpred=linpred)
        return errors
    
    def compute_residuals(self, pars):
        residuals_raw = self.compute_raw_residuals(pars)
        return self.noise_process.compute_residuals(pars, linpred=residuals_raw)
    
    def compute_raw_residuals(self, pars):
        
        # Time array
        t = self.data.t
        
        # The raw data rvs
        data_arr = self.data.get_trainable()
        
        # Remove the offsets from the data
        data_arr -= self.trend_model.build_trend_zero(pars, t)
        
        # Build the Keplerian model + trend
        model_arr = self.builder(pars, t)
        
        # Residuals
        residuals = data_arr - model_arr
        
        return residuals
    
    def __repr__(self):
        return "Composite GP RV Model"
        

import pychell.orbits.orbitbases as orbitbases