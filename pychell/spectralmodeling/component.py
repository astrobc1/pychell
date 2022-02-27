# Base Python
import glob

# Maths
import numpy as np
from scipy.special import eval_legendre
import scipy.interpolate

# pychell
import pychell.maths as pcmath
import pychell.utils as pcutils

# Optimize
from optimize.parameters import BoundedParameters, BoundedParameter


####################
#### BASE TYPES ####
####################

class SpectralModelComponent1d:
    """Base class for a general spectral component model.
    """

    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self):

        # No parameter names, probably overwritten with each instance
        self.par_names = []
        
    def get_init_parameters(self, data, templates, sregion):
        return BoundedParameters()

    def lock_parameters(self, pars):
        for pname in self.par_names:
            pars[pname].vary = False
            
    def vary_parameters(self, pars):
        for pname in self.par_names:
            pars[pname].vary = True

    ###############
    #### MISC. ####
    ###############

    def __repr__(self):
        s = f"Spectral model: {self.__class__.__name__}\n"
        for pname in self.par_names:
            s += f" {pname}\n"
        return s