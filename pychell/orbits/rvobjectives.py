import optimize.objectives as optobj
import optimize.noise as optnoise
import time
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit

class RVLikelihood(optobj.Likelihood):
    
    def __init__(self, data, model, noise, p0=None, label=None):
        super().__init__(data=data, model=model, noise=noise, p0=p0, label=label)
        self.data_t = self.data.gen_vec("t")
        self.data_rv = self.data.gen_vec("rv")
        self.data_rverr = self.data.gen_vec("rverr")
    
    def compute_logL(self, pars):
        """Computes the log of the likelihood.
    
        Args:
            pars (Parameters): The parameters to use.
            apply_priors (bool, optional): Whether or not to apply the priors. Defaults to True.

        Returns:
            float: The log likelihood, ln(L).
        """

        # Get residuals
        residuals = self.compute_data_pre_noise_process(pars)

        # Compute the cov matrix
        K = self.noise.compute_cov_matrix(pars, include_uncorr_error=True)
        
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
    
    def compute_data_pre_noise_process(self, pars):
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
    
    def compute_data_post_noise_process(self, pars):
        """Computes the residuals after subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        
        # Get the data containing only noise
        data_pre_noise_process = self.compute_data_pre_noise_process(pars)
        
        # Copy the data
        data_post_noise_process = np.copy(data_pre_noise_process)
        
        # If noise is correlated, the mean may not be zero, so realize the noise process and subtract.
        if isinstance(self.noise, optnoise.CorrelatedNoiseProcess):
            noise_process_mean = self.noise.realize(pars, data_pre_noise_process)
            data_post_noise_process -= noise_process_mean
        return data_post_noise_process
    
    def get_components(self, pars):
        
        comps = {}
        
        # Data times
        data_t = np.copy(self.data_t)
        
        # Data RVs - trends
        data_rvs = np.copy(self.data_rv)
        data_rvs -= self.model.apply_offsets(pars, data_t, instname=None)
        
        # Get residuals
        residuals_with_noise = self.compute_data_pre_noise_process(pars)
        
        # Data errrors
        data_rvs_error = self.noise.compute_data_errors(pars, include_gp_error=True, data_with_noise=residuals_with_noise)
        
        # Store in comps
        comps[self.label + "_data_t"] = data_t
        comps[self.label + "_data_rvs"] = data_rvs
        comps[self.label + "_data_rvs_error"] = data_rvs_error
        
        # Standard GP
        if isinstance(self.noise, optnoise.CorrelatedNoiseProcess):
            gp_mean, gp_unc = self.noise.realize(pars, data_with_noise=residuals_with_noise, xpred=data_t, return_gp_error=True)
            comps[self.label + "_gp_mean"] = gp_mean
            comps[self.label + "_gp_unc"] = gp_unc

        return comps

    def gen_inds(self):
        inds = {}
        for like in self.values():
            t = np.concatenate((t, like.data_t))
        ss = np.argsort(t)
        return ss


class RVChromaticLikelihoodJ1(RVLikelihood):
    
    def compute_data_post_noise_process(self, pars):
        """Computes the residuals after subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        residuals_with_noise = self.compute_data_pre_noise_process(pars)
        residuals_no_noise = np.copy(residuals_with_noise)
        for data in self.data.values():
            inds = self.model.data_inds[data.label]
            gp_mean = self.noise.realize(pars, data_with_noise=residuals_with_noise, xpred=data.t, return_gp_error=False, instname=data.label)
            residuals_no_noise[inds] -= gp_mean
            
        return residuals_no_noise
    
    def get_components(self, pars):
        
        comps = {}
        
        # Data times
        data_t = np.copy(self.data_t)
        
        # Data RVs - trends
        data_rvs = np.copy(self.data_rv)
        data_rvs = self.model.apply_offsets(data_rvs, pars, data_t, instname=None)
        
        # Get residuals
        residuals_with_noise = self.compute_data_pre_noise_process(pars)
        
        # Data errrors
        data_rvs_error = self.noise.compute_data_errors(pars, include_gp_error=True, data_with_noise=residuals_with_noise)
        
        # Store in comps
        comps[self.label + "_data_t"] = data_t
        comps[self.label + "_data_rvs"] = data_rvs
        comps[self.label + "_data_rvs_error"] = data_rvs_error
        
        # Standard GP
        for data in self.data.values():
            gp_mean, gp_unc = self.noise.realize(pars, data_with_noise=residuals_with_noise, xpred=data.t, xdata=None, return_gp_error=True, instname=data.label)
            comps[self.label + "_gp_mean_" + data.label] = gp_mean
            comps[self.label + "_gp_unc_" + data.label] = gp_unc

        return comps

class RVChromaticLikelihoodJ2(RVLikelihood):
    
    def compute_data_post_noise_process(self, pars):
        """Computes the residuals after subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        residuals_with_noise = self.compute_data_pre_noise_process(pars)
        residuals_no_noise = np.copy(residuals_with_noise)
        for data in self.data.values():
            inds = self.model.data_inds[data.label]
            gp_mean = self.noise.realize(pars, data_with_noise=residuals_with_noise, xpred=data.t, return_gp_error=False, wavelength=data.wavelength)
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

class RVChromaticLikelihoodJ3(RVLikelihood):
    
    def compute_data_post_noise_process(self, pars):
        """Computes the residuals after subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        residuals_with_noise = self.compute_data_pre_noise_process(pars)
        residuals_no_noise = np.copy(residuals_with_noise)
        for data in self.data.values():
            inds = self.model.data_inds[data.label]
            gp_mean = self.noise.realize(pars, data_with_noise=residuals_with_noise, xpred=data.t, return_gp_error=False, instname=data.label)
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
    """A class for RV Posteriors
    """
    
    def compute_redchi2(self, pars, include_gp_error=False, gp_error=None):
        
        residuals = np.array([], dtype=float)
        errors = np.array([], dtype=float)
        for like in self.values():
            residuals_with_noise = like.compute_data_pre_noise_process(pars)
            residuals_no_noise = like.compute_data_post_noise_process(pars)
            errs = like.noise.compute_data_errors(pars, include_gp_error=include_gp_error, gp_error=gp_error, data_with_noise=residuals_with_noise)
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
        
        # All data
        comps["data_t"] = np.copy(self.data_t)
        comps["data_rv"] = np.copy(self.data_rv)
        comps["data_rverr"] = np.copy(self.data_rverr)
        
        # Planets
        for planet_index in self.like0.model.planets_dict:
            for like in self.likes():
                planet_signal = self.like0.model.build_planet(pars, comps["data_t"], planet_index)
                comps["planet_" + str(planet_index) + "_rvs"] = planet_signal
            
        # Data and GP
        for like in self.values():
            _comps = like.get_components(pars)
            comps.update(_comps)
        
        return comps
    
    @property
    def data_t(self):
        t = np.array([], dtype=float)
        for like in self.values():
            t = np.concatenate((t, like.data_t))
        ss = np.argsort(t)
        t = t[ss]
        return t
    
    @property
    def data_rv(self):
        rv = np.array([], dtype=float)
        for like in self.values():
            rv = np.concatenate((rv, like.data_rv))
        return rv
    
    @property
    def data_rverr(self):
        rverr = np.array([], dtype=float)
        for like in self.values():
            rverr = np.concatenate((rverr, like.data_rverr))
        rverr = rverr[self.like_inds]
        return rverr

    
  
    @property
    def instname_vec(self):
        instnames = np.array([], dtype=float)
        t = np.array([], dtype=float)
        for like in self.values():
            instnames = np.concatenate((instnames, like.data.gen_tel_vec()))
            t = np.concatenate(())
        #instnames = instnames[]
        return instnames
    
    @property
    def data_inds(self):
        instnames = self.instnames_vec
        sorting_inds = {}
        for like in self.likes():
            for data in like.data.values():
                inds = np.where(data.label == self.instnames)
                sorting_inds[data.label] = inds
        return sorting_inds