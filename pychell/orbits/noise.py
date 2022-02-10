import numpy as np
from scipy.linalg import cho_solve, cho_factor
import matplotlib.pyplot as plt
import optimize.noise as optnoise
from numba import njit, prange

# Import optimize kernels into namespace
from optimize.noise import WhiteNoiseProcess, GaussianProcess
from optimize.kernels import CorrelatedNoiseKernel

# Import rv kernels into namespace
from pychell.orbits.kernels import ChromaticKernelJ1, ChromaticKernelJ2

class ChromaticProcessJ1(GaussianProcess):
    
    def __init__(self, par_names=None):
        super().__init__(kernel=ChromaticKernelJ1(par_names=par_names))

    def compute_cov_matrix(self, pars, x1, x2, amp_vec1, amp_vec2, data_errors=None, include_uncorrelated_error=True):

        # Get K
        K = self.kernel.compute_cov_matrix(pars, x1, x2, amp_vec1, amp_vec2)

        # Data errors
        if include_uncorrelated_error:
            np.fill_diagonal(K, np.diag(K) + data_errors**2)
        
        return K
        
    def predict(self, pars, linpred, amp_vec_data, amp, xdata, xpred=None, data_errors=None):
        
        # Get grids
        if xpred is None:
            xpred = xdata
        
        # Get K
        amp_vec_pred = np.full(len(xpred), amp)
        K = self.compute_cov_matrix(pars, xdata, xdata, amp_vec_data, amp_vec_data, data_errors=data_errors, include_uncorrelated_error=True)
        
        # Compute version of K without intrinsic data error
        Ks = self.compute_cov_matrix(pars, xpred, xdata, amp_vec_pred, amp_vec_data, data_errors=None, include_uncorrelated_error=False)

        # Avoid overflow errors by reducing the matrix.
        L = cho_factor(K)

        # Solve
        alpha = cho_solve(L, linpred)
        mu = np.dot(Ks, alpha).flatten()

        # Compute uncertainty
        Kss = self.compute_cov_matrix(pars, xpred, xpred, amp_vec_pred, amp_vec_pred, data_errors=None, include_uncorrelated_error=False)
        B = cho_solve(L, Ks.T)
        error = np.sqrt(np.array(np.diag(Kss - np.dot(Ks, B))).flatten())

        # Return
        return mu, error


class ChromaticProcessJ2(GaussianProcess):
    
    def __init__(self, par_names=None, wavelength0=550):
        super().__init__(kernel=ChromaticKernelJ2(par_names=par_names, wavelength0=wavelength0))

    def compute_cov_matrix(self, pars, x1, x2, wave_vec1, wave_vec2, data_errors=None, include_uncorrelated_error=True):

        # Get K
        K = self.kernel.compute_cov_matrix(pars, x1, x2, wave_vec1, wave_vec2)

        # Data errors
        if include_uncorrelated_error:
            np.fill_diagonal(K, np.diag(K) + data_errors**2)

        return K
        
    def predict(self, pars, linpred, wave_vec_data, wave, xdata, xpred=None, data_errors=None):
        
        # Get grids
        if xpred is None:
            xpred = xdata
        
        # Get K
        wave_vec_pred = np.full(len(xpred), wave)
        K = self.compute_cov_matrix(pars, xdata, xdata, wave_vec_data, wave_vec_data, data_errors=data_errors, include_uncorrelated_error=True)
        
        # Compute version of K without intrinsic data error
        Ks = self.compute_cov_matrix(pars, xpred, xdata, wave_vec_pred, wave_vec_data, data_errors=None, include_uncorrelated_error=False)

        # Avoid overflow errors by reducing the matrix.
        L = cho_factor(K)

        # Solve
        alpha = cho_solve(L, linpred)
        mu = np.dot(Ks, alpha).flatten()

        # Compute uncertainty
        Kss = self.compute_cov_matrix(pars, xpred, xpred, wave_vec_pred, wave_vec_pred, data_errors=None, include_uncorrelated_error=False)
        B = cho_solve(L, Ks.T)
        error = np.sqrt(np.array(np.diag(Kss - np.dot(Ks, B))).flatten())

        # Return
        return mu, error