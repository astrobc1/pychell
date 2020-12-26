import optimize.scores as optscore
import optimize.kernels as optnoisekernels
import time
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit

class RVLikelihood(optscore.Likelihood):
    
    def __init__(self, label=None, data=None, model=None):
        super().__init__(label=label, data=data, model=model)
        self.p0 = model.p0
        self.data_inds = {}
        for data in self.data.values():
            self.data_inds[data.label] = self.data.get_inds(data.label)
    
    def compute_logL(self, pars, apply_priors=True):
        """Computes the log of the likelihood.
    
        Args:
            pars (Parameters): The parameters to use.
            apply_priors (bool, optional): Whether or not to apply the priors. Defaults to True.

        Returns:
            float: The log likelihood, ln(L).
        """
        # Apply priors, see if we even need to compute the model
        if apply_priors:
            lnL = self.compute_logL_priors(pars)
            if not np.isfinite(lnL):
                return -np.inf
        else:
            lnL = 0

        # Copy the rvs for this likelihood
        data_arr = np.copy(self.data_rv)
        
        # Compute the model (consistent across all data sets for this likelihood).
        model_arr = self.model.build(pars)
        
        # Apply offsets
        data_arr = self.model.apply_offsets(data_arr, pars)

        # Compute the residuals for this data group
        residuals = data_arr - model_arr

        # Compute the cov matrix
        K = self.model.kernel.compute_cov_matrix(pars, apply_errors=True)

        # Compute the determiniant and inverse of K
        try:
        
            # Reduce the cov matrix and solve for KX = residuals
            alpha = cho_solve(cho_factor(K), residuals)

            # Compute the log determinant of K
            _, lndetK = np.linalg.slogdet(K)

            # Compute the likelihood
            N = len(data_arr)
            lnL += -0.5 * (np.dot(residuals, alpha) + lndetK + N * np.log(2 * np.pi))
    
        except:
            # If things fail (matrix decomp) return -inf
            lnL = -np.inf
        
        # Return the final ln(L)
        return lnL
    
    def residuals_before_kernel(self, pars):
        """Computes the residuals without subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        model_arr = self.model.build(pars)
        data_arr = np.copy(self.data_rv)
        data_arr = self.model.apply_offsets(data_arr, pars)
        residuals = data_arr - model_arr
        return residuals
    
    def residuals_after_kernel(self, pars):
        """Computes the residuals after subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        residuals = self.residuals_before_kernel(pars)
        if not self.model.kernel.is_diag:
            kernel_mean = self.model.kernel.realize(pars, residuals, return_unc=False)
            residuals -= kernel_mean
        return residuals
    
    def data_only_planet(self, pars, planet_index):
        """Removes the full model from the data except for one planet.

        Args:
            pars (Parameters): The parameters.
            planet_index (int): The planet index to keep in the data.

        Returns:
            dict: The modified data as a dictionary, where keys are the labels, and values are numpy arrays.
        """
        mod_data = {}
        residuals = self.residuals_after_kernel(pars)
        planet_model_arr = self.model.build_planet(pars, self.data_t, planet_index)
        for data in self.data.values():
            mod_data[data.label] = residuals[self.data_inds[data.label]] + planet_model_arr[self.data_inds[data.label]]
        
        return mod_data
    
    @property
    def data_t(self):
        return self.data_x
    
    @property
    def data_rv(self):
        return self.data_y
    
    @property
    def data_rverr(self):
        return self.data_yerr
    
class RVChromaticLikelihood(RVLikelihood):
    
    def residuals_after_kernel(self, pars):
        """Computes the residuals after subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        residuals = self.residuals_before_kernel(pars)
        for data in self.data.values():
            gp_mean = self.model.kernel.realize(pars, residuals=residuals[self.data_inds[data.label]], xres=data.t, instname=data.label, return_unc=False)
            residuals[self.model.data_inds[data.label]] -= gp_mean
        return residuals
    
    
class MixedRVLikelihood(optscore.MixedLikelihood):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redchi2s = []
    
    def compute_logL(self, pars, apply_priors=True):
        """Computes the log of the likelihood.
    
        Args:
            pars (Parameters): The parameters to use.
            apply_priors (bool, optional): Whether or not to apply the priors. Defaults to True.

        Returns:
            float: The log likelihood, ln(L).
        """
        #self.redchi2s.append(self.compute_redchi2(pars))
        lnL = 0
        if apply_priors:
            lnL += self.compute_logL_priors(pars)
            if not np.isfinite(lnL):
                return -np.inf
        for like in self.values():
            lnL += like.compute_logL(pars, apply_priors=False)
        return lnL