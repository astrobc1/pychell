import optimize.objectives as optobj
import optimize.noise as optnoise
import time
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
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
        K = self.noise.compute_cov_matrix(pars)
        
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
        
        # Data times for this label
        comps[self.label + "_data_t"] = np.copy(self.data_t)
        
        # Data RVs - trends
        data_rvs = np.copy(self.data_rv)
        comps[self.label + "_data_rvs"] = self.model.apply_offsets(data_rvs, pars, instname=None)
        
        # Get residuals
        residuals_with_noise = self.compute_data_pre_noise_process(pars)
        
        # Intrinsic + Jitter errors
        comps[self.label + "_data_rvs_error"] = self.noise.compute_data_errors(pars, include_gp_error=False, data_with_noise=residuals_with_noise)
        
        # Planets
        for planet_index in self.model.planets_dict:
            comps[self.label + "_planet_" + str(planet_index)] = self.model.build_planet(pars, comps[self.label + "_data_t"], planet_index)
        
        # Standard GP
        if isinstance(self.noise, optnoise.CorrelatedNoiseProcess):
            gp_mean, gp_unc = self.noise.realize(pars, data_with_noise=residuals_with_noise, xpred=comps[self.label + "_data_t"], return_gp_error=True)
            comps[self.label + "_gp_mean"] = gp_mean
            comps[self.label + "_gp_unc"] = gp_unc

        return comps


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
        
        # Planets
        for planet_index in self.model.planets_dict:
            comps[self.label + "_planet_" + str(planet_index)] = self.model.build_planet(pars, data_t, planet_index)
        
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
        data_rvs_error = self.kernel.compute_data_errors(pars, include_gp_error=True, residuals_with_noise=residuals_with_noise)
        
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
    """A class for RV Posteriors.
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
            for like in self.values():
                planet_signal = self.like0.model.build_planet(pars, comps["data_t"], planet_index)
                comps["planet_" + str(planet_index) + "_rvs"] = planet_signal
            
        # Data and GP
        for like in self.values():
            _comps = like.get_components(pars)
            comps.update(_comps)
        
        return comps
    
    def gen_label_vec(self):
        
        # Times
        t = np.array([], dtype=float)
        for like in self.values():
            t = np.concatenate((t, like.data_t))
        
        # Labels
        label_vec = np.array([], dtype='<U50')
        for like in self.values():
            label_vec = np.concatenate((label_vec, np.full(len(like.data_t), fill_value=like.label, dtype='<U50')))
            
        ss = np.argsort(t)
        label_vec = label_vec[ss]
        return label_vec
    
    def gen_wave_vec(self):
        t = np.array([], dtype=float)
        waves = np.array([], dtype=float)
        for like in self.values():
            waves = np.concatenate((waves, like.data.gen_wave_vec()))
            t = np.concatenate((t, like.data_t))
        ss = np.argsort(t)
        waves = waves[ss]
        return waves
    
    def gen_like_inds(self):
        like_label_vec = self.gen_label_vec()
        inds = {}
        for like in self.values():
            inds[like.label] = np.where(like_label_vec == like.label)[0]
        return inds
    
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
        t = np.array([], dtype=float)
        rv = np.array([], dtype=float)
        for like in self.values():
            rv = np.concatenate((rv, like.data_rv))
            t = np.concatenate((t, like.data_t))
        ss = np.argsort(t)
        rv = rv[ss]
        return rv
    
    @property
    def data_rverr(self):
        t = np.array([], dtype=float)
        rverr = np.array([], dtype=float)
        for like in self.values():
            t = np.concatenate((t, like.data_t))
            rverr = np.concatenate((rverr, like.data_rverr))
        ss = np.argsort(t)
        rverr = rverr[ss]
        return rverr


    def __setitem__(self, label, like):
        super().__setitem__(label, like)
        self.like_inds = self.gen_like_inds()

class ChromaticPosterior(RVPosterior):
    
    def __init__(self, *args, sep=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.sep = sep
        self.color_obs = self.get_color_obs()
            
    def compute_logaprob(self, pars):
        
        # Compute default logL
        lnL = super().compute_logL(pars)
        
        # Add rv color information
        try:
            lnL += self.compute_rvcolor_L2(pars)
        except:
            pass
        return lnL
        
    def compute_rvcolor_L2(self, pars):
        data_color = np.zeros(len(self.color_obs))
        data_color_unc = np.zeros(len(self.color_obs))
        gp_color = np.zeros(len(self.color_obs))
        L = pars["lambda"].value
        data_rv = self.data_rv
        data_rverr = np.zeros_like(self.data_t)
        data_with_noise = {}
        for like in self.values():
            inds = self.like_inds[like.label]
            data_rv[inds] = like.model.apply_offsets(data_rv[self.like_inds[like.label]], pars)
            data_rverr[inds] = like.noise.compute_data_errors(pars)
            data_with_noise[like.label] = like.compute_data_pre_noise_process(pars)
        for i, cobs in enumerate(self.color_obs):
            t = np.atleast_1d(cobs["t"])
            inds = cobs["inds"]
            labels = cobs["like_labels"]
            ind1, ind2 = inds[0], inds[1]
            label1, label2 = labels[0], labels[1]
            data_color[i] = data_rv[ind2] - data_rv[ind1]
            data_color_unc[i] = np.sqrt(data_rverr[ind1]**2 + data_rverr[ind2]**2)
            gp1 = self[label1].noise.realize(pars, data_with_noise=data_with_noise[label1], xdata=None, xpred=t, return_gp_error=False)
            gp2 = self[label2].noise.realize(pars, data_with_noise=data_with_noise[label2], xdata=None, xpred=t, return_gp_error=False)
            gp_color[i] = gp1[0] - gp2[0]
        residuals_color = data_color - gp_color
        n = len(residuals_color)
        chi2 = (1 / n) * np.nansum((residuals_color / data_color_unc)**2)
        L2reg = - L * chi2 / 2
        return L2reg
    
    def get_color_obs(self):
        
        # Get the relevant full data vectors
        data_t = np.copy(self.data_t)
        like_label_vec = self.gen_label_vec()
        wave_vec = self.gen_wave_vec()

        # Loop over RVs and look for near-simultaneous RV color observations.
        color_obs = []
        prev_i = 0
        n_data = len(data_t)
        for i in range(n_data - 1):
            
            # If dt > sep, we have moved to the next night.
            # But first ook at all RVs from this night
            if data_t[i+1] - data_t[i] > self.sep:
                
                # The number of RVs on this night for these two wavelengths.
                n_obs_night = i - prev_i + 1
                
                # If only 1 observation for this night, skip it.
                if n_obs_night < 2:
                    prev_i = i + 1
                    continue
                
                # The indices for this night, relative to the filtered arrays 
                inds_this_night = np.arange(prev_i, i + 1).astype(int)
                
                # The wavelengths for this night
                waves_this_night = wave_vec[inds_this_night]
                
                # Ensure we have at least 2 unique wavelengths
                if len(np.unique(waves_this_night)) == 1:
                    prev_i = i + 1
                    continue
                
                # The like labels for each observation to generate the gps later on
                like_labels_this_night = like_label_vec[inds_this_night]
                
                # Get unique pairs of color observations
                for k in range(len(waves_this_night)):
                    for j in range(len(waves_this_night)):
                        if waves_this_night[k] == waves_this_night[j]:
                            continue
                        if waves_this_night[k] < waves_this_night[j]:
                            waves = (waves_this_night[k], waves_this_night[j])
                            inds = (inds_this_night[k], inds_this_night[j])
                            like_labels = (like_labels_this_night[k], like_labels_this_night[j])
                        else:
                            waves = (waves_this_night[j], waves_this_night[k])
                            inds = (inds_this_night[j], inds_this_night[k])
                            like_labels = (like_labels_this_night[j], like_labels_this_night[k])

                        # Time
                        t = (data_t[inds[0]] + data_t[inds[1]]) / 2
                        
                        _dict = dict(waves=waves, inds=inds, like_labels=like_labels, t=t)
                    
                        # Add night to color info
                        if _dict not in color_obs:
                            color_obs.append(_dict)
                
                # Move on.
                prev_i = i + 1
        
        # Check last night.
        n_obs_night = n_data - prev_i
        
        # If only 1 observation for this night, skip it.
        if n_obs_night >= 2:
        
            # The indices for this night, relative to the filtered arrays 
            inds_this_night = np.arange(prev_i, len(data_t)).astype(int)
            
            # The wavelengths for this night
            waves_this_night = wave_vec[inds_this_night]
            
            # Ensure we have at least 2 unique wavelengths
            if len(np.unique(waves_this_night)) > 1:
            
                # The like labels for each observation to generate the gps later on
                like_labels_this_night = like_label_vec[inds_this_night]
            
                # Get unique pairs of color observations
                for k in range(len(waves_this_night)):
                    for j in range(len(waves_this_night)):
                        if waves_this_night[k] == waves_this_night[j]:
                            continue
                        if waves_this_night[k] < waves_this_night[j]:
                            waves = (waves_this_night[k], waves_this_night[j])
                            inds = (inds_this_night[k], inds_this_night[j])
                            like_labels = (like_labels_this_night[k], like_labels_this_night[j])
                        else:
                            waves = (waves_this_night[j], waves_this_night[k])
                            inds = (inds_this_night[j], inds_this_night[k])
                            like_labels = (like_labels_this_night[j], like_labels_this_night[k])

                        # Time
                        t = (data_t[inds[0]] + data_t[inds[1]]) / 2
                        
                        _dict = dict(waves=waves, inds=inds, like_labels=like_labels, t=t)
                    
                        # Add night to color info
                        if _dict not in color_obs:
                            color_obs.append(_dict)
        
        return color_obs
    
    def __setitem__(self, label, like):
        super().__setitem__(label, like)
        self.color_obs = self.get_color_obs()