
# Maths
import numpy as np

# LLVM
import numba

# Pychell deps
import pychell.orbits.maths as planetmath

############################
#### KEPLERIAN RV MODEL ####
############################

class RVModel:
    
    #####################
    #### CONSTRUCTOR ####
    #####################
    
    def __init__(self, data, planets_dict=None, trend_poly_order=0, t0=None):
        self.data = data
        self.planets_dict = {} if planets_dict is None else planets_dict
        self.trend_poly_order = trend_poly_order
        self.t0 = np.nanmedian(data.t) if t0 is None else t0
        

    ##################
    #### BUILDERS ####
    ##################
    
    def build(self, pars, t=None):
        if t is None:
            t = self.data.t
        model = self.build_planets(pars, t)
        model += self.build_global_trend(pars, t)
        return model
         
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
        
        # Build
        vels = planetmath.planet_signal(t, *planet_pars)
        
        # Return
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
    
    def build_trend_zero(self, pars, t, instname=None):
        
        # Zeros
        trend_zero = np.zeros_like(t)
        
        if self.trend_poly_order is not None:
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
        if self.trend_poly_order > 0:
            for i in range(1, self.trend_poly_order + 1):
                trend += pars[f"gamma_{''.join(str(x) for x in (['d']*i))}ot"].value * (t - self.t0)**i
        return trend

    ###############
    #### MISC. ####
    ###############

    @staticmethod
    def disable_planet_pars(pars, planets_dict, planet_index):
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
        
    @property
    def n_planets(self):
        return len(self.planets_dict)

    def __repr__(self):
        return f"RV Model with {self.n_planets} planets"

from .bases import *