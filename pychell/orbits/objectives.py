import scipy
from scipy.linalg import cho_factor, cho_solve
import numpy as np
from optimize.bayesobjectives import GaussianLikelihood, Posterior
from optimize.noise import UnCorrelatedNoiseProcess
from pychell.orbits.noise import ChromaticProcessJ1, ChromaticProcessJ2

TWO_PI = 2 * np.pi
LOG_2PI = np.log(TWO_PI)

#######################
#### RV LIKELIHOOD ####
#######################

class RVLikelihood(GaussianLikelihood):

    def __init__(self, model=None, noise_process=None):
        self.model = model
        self.noise_process = noise_process

    def compute_cov_matrix(self, pars, include_uncorrelated_error=True):

        # Data errors
        data_errors = self.compute_data_errors(pars)

        # J1 kernel
        if type(self.noise_process) is ChromaticProcessJ1:
            data_amp_vec = np.full(self.datax.size, np.nan)
            data_instname_vec = self.model.data.gen_instname_vec()
            for i in range(len(data_instname_vec)):
                instname = data_instname_vec[i]
                data_amp_vec[i] = pars[f"gp_amp_{instname}"].value
            K = self.noise_process.compute_cov_matrix(pars, self.datax, self.datax, data_amp_vec, data_amp_vec, data_errors=data_errors, include_uncorrelated_error=include_uncorrelated_error)

        # J2 kernel
        elif type(self.noise_process) is ChromaticProcessJ2:
            data_wave_vec = self.model.data.gen_wave_vec()
            K = self.noise_process.compute_cov_matrix(pars, self.datax, self.datax, data_wave_vec, data_wave_vec, data_errors=data_errors, include_uncorrelated_error=include_uncorrelated_error)

        # Anything else
        else:
            K = self.noise_process.compute_cov_matrix(pars, self.datax, self.datax, data_errors=data_errors, include_uncorrelated_error=include_uncorrelated_error)

        return K

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

    def compute_data_errors(self, pars, include_correlated_error=False):

        # Intrinsic data errors
        errors2 = self.model.data.rverr**2

        # Additional white noise not already accounted for
        for instname in self.model.data:
            inds = self.model.data.indices[instname]
            errors2[inds] += pars[f"jitter_{instname}"].value**2

        # GP error
        if include_correlated_error:
            pass

        return np.sqrt(errors2)

    def compute_logL(self, pars):
        """Computes the log of the likelihood.
        
        Args:
            pars (Parameters): The parameters to use.

        Returns:
            float: The log likelihood, ln(L).
        """

        # Compute the residuals
        residuals = self.compute_residuals(pars)
        errors = self.compute_data_errors(pars)
        n = len(residuals)

        # Check if noise is correlated
        if isinstance(self.noise_process, UnCorrelatedNoiseProcess):
            lnL = -0.5 * (np.sum((residuals / errors)**2) + np.sum(np.log(errors**2)) + n * LOG_2PI)
            return lnL
        
        else:
        
            # Compute the determiniant and inverse of K
            try:

                # Compute the cov matrix
                K = self.compute_cov_matrix(pars, include_uncorrelated_error=True)

                # Reduce the cov matrix
                alpha = cho_solve(cho_factor(K), residuals)

                # Compute the log determinant of K
                _, lndetK = np.linalg.slogdet(K)

                # Compute the Gaussian likelihood
                lnL = -0.5 * (np.dot(residuals, alpha) + lndetK + n * LOG_2PI)

                # Return
                return lnL
        
            except scipy.linalg.LinAlgError:
                
                # If things fail, return -inf
                return -np.inf
    
    def compute_noise_components(self, pars, x):

        residuals = self.compute_residuals(pars)
        data_errors = self.compute_data_errors(pars)

        noise_components = {}

        # J1 kernel
        if type(self.noise_process) is ChromaticProcessJ1:
            data_amp_vec = np.full(self.datax.size, np.nan)
            data_instname_vec = self.model.data.gen_instname_vec()
            for i in range(len(data_instname_vec)):
                instname = data_instname_vec[i]
                data_amp_vec[i] = pars[f"gp_amp_{instname}"].value
            for instname in self.model.data:
                label = f"GP {instname}"
                amp = pars[f"gp_amp_{instname}"].value
                gp, gperr = self.noise_process.predict(pars, residuals, data_amp_vec, amp, self.datax, x, data_errors)
                inds = np.where(data_instname_vec == instname)[0]
                noise_components[label] = (gp, gperr, inds)

        # J2 kernel
        elif type(self.noise_process) is ChromaticProcessJ2:
            wave_vec_data = self.model.data.gen_wave_vec()
            waves = np.sort(np.unique(wave_vec_data))
            for wave in waves:
                label = f"GP {wave}"
                gp, gperr = self.noise_process.predict(pars, residuals, wave_vec_data, wave, self.datax, x, data_errors)
                inds = np.where(wave_vec_data == wave)[0]
                noise_components[label] = (gp, gperr, inds)

        # Anything else
        else:
            label = f"GP {list(self.model.data.keys())}"
            inds = np.arange(self.model.data.t.size).astype(int)
            gp, gperr = self.noise_process.predict(pars, residuals, self.datax, x, data_errors)
            noise_components = {label : (gp, gperr, inds)}

        return noise_components

    def __repr__(self):
        return "Gaussian Likelihood"

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