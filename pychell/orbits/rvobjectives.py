import optimize.objectives as optobj
import optimize.kernels as optnoisekernels
import time
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit

class RVLikelihood(optobj.Likelihood):
    
    def __init__(self, label=None, data=None, model=None, kernel=None):
        super().__init__(label=label, data=data, model=model, kernel=kernel)
        self.data_inds = {}
        for data in self.data.values():
            self.data_inds[data.label] = self.data.get_inds(data.label)
    
    def compute_logL(self, pars):
        """Computes the log of the likelihood.
    
        Args:
            pars (Parameters): The parameters to use.
            apply_priors (bool, optional): Whether or not to apply the priors. Defaults to True.

        Returns:
            float: The log likelihood, ln(L).
        """

        # Get residuals
        residuals = self.residuals_with_noise(pars)

        # Compute the cov matrix
        K = self.kernel.compute_cov_matrix(pars, include_white_error=True)
        
        # Compute the determiniant and inverse of K
        try:
            # Reduce the cov matrix and solve for KX = residuals
            alpha = cho_solve(cho_factor(K), residuals)

            # Compute the log determinant of K
            _, lndetK = np.linalg.slogdet(K)

            # Compute the likelihood
            n = len(residuals)
            lnL = -0.5 * (np.dot(residuals, alpha) + lndetK + n * np.log(2 * np.pi))
    
        except:
            # If things fail (matrix decomp) return -inf
            lnL = -np.inf
        
        # Return the final ln(L)
        return lnL
    
    def residuals_with_noise(self, pars):
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
    
    def residuals_no_noise(self, pars):
        """Computes the residuals after subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        residuals_with_noise = self.residuals_with_noise(pars)
        residuals_no_noise = np.copy(residuals_with_noise)
        if isinstance(self.kernel, optnoisekernels.CorrelatedNoiseKernel):
            kernel_mean = self.kernel.realize(pars, residuals_with_noise, return_kernel_error=False, kernel_error=None)
            residuals_no_noise -= kernel_mean
        return residuals_no_noise
    
    def get_components(self, pars):
        
        comps = {}
        
        # Data times
        data_t = np.copy(self.data_t)
        
        # Data RVs - offsets
        data_rvs = np.copy(self.data_rv)
        data_rvs -= self.model.build_trend_zero(pars, data_t, instname=None)
        
        # Get residuals
        residuals_with_noise = self.residuals_with_noise(pars)
        
        # Data errrors
        data_rvs_error = self.kernel.compute_data_errors(pars, include_white_error=True, include_kernel_error=True, residuals_with_noise=residuals_with_noise)
        
        # Store in comps
        comps[self.label + "_data_t"] = data_t
        comps[self.label + "_data_rvs"] = data_rvs
        comps[self.label + "_data_rvs_error"] = data_rvs_error
        
        # Standard GP
        if isinstance(self.kernel, optnoisekernels.CorrelatedNoiseKernel):
            kernel_mean, kernel_unc = self.kernel.realize(pars, residuals_with_noise=residuals_with_noise, xpred=data_t, xres=None, return_kernel_error=True)
            comps[self.label + "_kernel_mean"] = kernel_mean
            comps[self.label + "_kernel_unc"] = kernel_unc

        return comps
    
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
    
    def residuals_no_noise(self, pars):
        """Computes the residuals after subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        residuals_with_noise = self.residuals_with_noise(pars)
        residuals_no_noise = np.copy(residuals_with_noise)
        for data in self.data.values():
            inds = self.model.data_inds[data.label]
            gp_mean = self.kernel.realize(pars, residuals_with_noise=residuals_with_noise, xpred=data.t, wavelength=data.wavelength, return_kernel_error=False)
            residuals_no_noise[inds] -= gp_mean
            
        return residuals_no_noise
    
    def get_components(self, pars):
        
        comps = {}
        
        # Data times
        data_t = np.copy(self.data_t)
        
        # Data RVs - offsets
        data_rvs = np.copy(self.data_rv)
        data_rvs -= self.model.build_trend_zero(pars, data_t, instname=None)
        
        # Get residuals
        residuals_with_noise = self.residuals_with_noise(pars)
        
        # Data errrors
        data_rvs_error = self.kernel.compute_data_errors(pars, include_white_error=True, include_kernel_error=True, residuals_with_noise=residuals_with_noise)
        
        # Store in comps
        comps[self.label + "_data_t"] = data_t
        comps[self.label + "_data_rvs"] = data_rvs
        comps[self.label + "_data_rvs_error"] = data_rvs_error
        
        # Standard GP
        for data in self.data.values():
            kernel_mean, kernel_unc = self.kernel.realize(pars, residuals_with_noise=residuals_with_noise, xpred=data.t, xres=None, return_kernel_error=True, wavelength=data.wavelength)
            comps[self.label + "_kernel_mean_" + data.label] = kernel_mean
            comps[self.label + "_kernel_unc_" + data.label] = kernel_unc
        
        return comps
    
class RVChromaticLikelihood2(RVLikelihood):
    
    def residuals_no_noise(self, pars):
        """Computes the residuals after subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        residuals_with_noise = self.residuals_with_noise(pars)
        residuals_no_noise = np.copy(residuals_with_noise)
        for data in self.data.values():
            inds = self.model.data_inds[data.label]
            gp_mean = self.kernel.realize(pars, residuals_with_noise=residuals_with_noise, xpred=data.t, return_kernel_error=False, instrument=data.label)
            residuals_no_noise[inds] -= gp_mean
            
        return residuals_no_noise
    
    def get_components(self, pars):
        
        comps = {}
        
        # Data times
        data_t = np.copy(self.data_t)
        
        # Data RVs - offsets
        data_rvs = np.copy(self.data_rv)
        data_rvs = self.model.apply_offsets(data_rvs, pars)
        
        # Get residuals
        residuals_with_noise = self.residuals_with_noise(pars)
        
        # Data errrors
        data_rvs_error = self.kernel.compute_data_errors(pars, include_white_error=True, include_kernel_error=True, residuals_with_noise=residuals_with_noise)
        
        # Store in comps
        comps[self.label + "_data_t"] = data_t
        comps[self.label + "_data_rvs"] = data_rvs
        comps[self.label + "_data_rvs_error"] = data_rvs_error
        
        # Standard GP
        for data in self.data.values():
            kernel_mean, kernel_unc = self.kernel.realize(pars, residuals_with_noise=residuals_with_noise, xpred=data.t, xres=None, return_kernel_error=True, instrument=data.label)
            comps[self.label + "_kernel_mean_" + data.label] = kernel_mean
            comps[self.label + "_kernel_unc_" + data.label] = kernel_unc
        
        return comps

class RVPosterior(optobj.Posterior):
    """Probably identical to Posterior.
    """
    
    def compute_redchi2(self, pars, include_white_error=True, include_kernel_error=True, kernel_error=None):
        
        residuals = np.array([], dtype=float)
        errors = np.array([], dtype=float)
        for like in self.values():
            residuals_with_noise = like.residuals_with_noise(pars)
            residuals_no_noise = like.residuals_no_noise(pars)
            errs = like.kernel.compute_data_errors(pars, include_white_error=include_white_error, include_kernel_error=include_kernel_error, kernel_error=kernel_error, residuals_with_noise=residuals_with_noise)
            residuals = np.concatenate((residuals, residuals_no_noise))
            errors = np.concatenate((errors, errs))
        
        # Compute red chi2, no need to sort.
        n_data = len(residuals)
        n_pars_vary = pars.num_varied()
        n_dof = n_data - n_pars_vary
        assert n_dof > 0
        redchi2 = np.nansum((residuals / errors)**2) / n_dof
        return redchi2
    
    def get_components(self, pars):
        
        # Components
        comps = {}
        
        # Time vector for this likelihood
        t_vec = np.copy(self.data_t)
        
        # Planets
        for planet_index in self.like0.model.planets_dict:
            planet_signal = self.like0.model.build_planet(pars, t_vec, planet_index)
            comps["planet_" + str(planet_index) + "_rvs"] = planet_signal
            
        # Data and GP
        for like in self.values():
            _comps = like.get_components(pars)
            comps.update(_comps)
        
        return comps
    
    @property
    def data_t(self):
        data_t = np.array([], dtype=float)
        for like in self.values():
            data_t = np.concatenate((data_t, like.data_t))
        ss = np.argsort(data_t)
        data_t = data_t[ss]
        return data_t
    
    @property
    def data_rv(self):
        data_rv = np.array([], dtype=float)
        for like in self.values():
            data_rv = np.concatenate((data_rv, like.data_rv))
        data_rv = data_rv[self.sorting_inds]
        return data_rv
    
    @property
    def data_rverr(self):
        data_rverr = np.array([], dtype=float)
        for like in self.values():
            data_rverr = np.concatenate((data_rverr, like.data_rverr))
        data_rverr = data_rverr[self.sorting_inds]
        return data_rverr
    
    @property
    def sorting_inds(self):
        data_t = np.array([], dtype=float)
        for like in self.values():
            data_t = np.concatenate((data_t, like.data_t))
        ss = np.argsort(data_t)
        return ss