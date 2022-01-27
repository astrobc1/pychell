from scipy.linalg import cho_factor, cho_solve
import numpy as np
from optimize.bayesobjectives import GaussianLikelihood, Posterior

#######################
#### RV LIKELIHOOD ####
#######################

class RVLikelihood(GaussianLikelihood):

    def __init__(self, model=None, noise_process=None):
        self.model = model
        self.noise_process = noise_process

    def compute_residuals(self, pars):
        
        # Time array
        t = self.model.data.t
        
        # The raw data rvs
        data = self.model.data.rv
        
        # Build the Keplerian + trend model
        model = self.model.build(pars, t) + self.model.build_trend_zero(pars, t)
        
        # Residuals
        residuals = data - model
        
        # Return
        return residuals

    def compute_data_errors(self, pars):
        errors2 = self.model.data.rverr**2
        for instname in self.model.data:
            inds = self.model.data.indices[instname]
            errors2[inds] += pars[f"jitter_{instname}"].value**2
        return np.sqrt(errors2)

    @property
    def datax(self):
        return self.model.data.t

    @property
    def datay(self):
        return self.model.data.rv
    
    @property
    def datayerr(self):
        return self.model.data.rverr

    def __repr__(self):
        s = "RV Likelihood:\n"
        s += " Data:\n"
        s += f" {self.model.data}\n"
        s += " Model:\n"
        s += f" {self.model}:"
        return s

######################
#### RV POSTERIOR ####
######################

# class RVPosterior(Posterior):
#     """A class for RV Posteriors.
#     """
#     pass