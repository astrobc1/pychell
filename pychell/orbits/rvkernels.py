import numpy as np
from scipy.linalg import cho_solve, cho_factor
import matplotlib.pyplot as plt
import optimize.kernels as optkernels
from numba import njit, prange

# Import optimize kernels into namespace
from optimize.kernels import *

class RVColor(optkernels.GaussianProcess):
    
    def __init__(self, data, par_names=None, wavelength0=550):
        super().__init__(data=data, par_names=par_names)
        self.tel_vec = self.data.make_tel_vec()
        self.wavelength0 = wavelength0
        self.unique_wavelengths = self.make_wave_vec_unique()
        self.compute_dist_matrix()
        
    def compute_cov_matrix(self, pars, include_white_error=True):
        
        # Alias params
        eta1 = pars[self.par_names[0]].value # amp linear wave scale
        eta2 = pars[self.par_names[1]].value # amp power law wave scale
        eta3 = pars[self.par_names[2]].value # decay
        eta4 = pars[self.par_names[3]].value # period
        eta5 = pars[self.par_names[4]].value # smoothing factor
            
        # Construct individual kernels
        lin_kernel = (eta1 * self.freq_matrix**eta2)**2
        decay_kernel = np.exp(-0.5 * (self.dist_matrix / eta3)**2)
        periodic_kernel = np.exp(-0.5 * (1 / eta5)**2 * np.sin(np.pi * self.dist_matrix / eta4)**2)
        
        # Construct full cov matrix
        cov_matrix = lin_kernel * decay_kernel * periodic_kernel
        
        # Apply intrinsic plus white noise data errors
        if include_white_error:
            data_errors = self.compute_data_errors(pars, include_white_error=include_white_error, include_kernel_error=False)
            np.fill_diagonal(cov_matrix, np.diag(cov_matrix) + data_errors**2)
        
        return cov_matrix
                    
    def compute_data_errors(self, pars, include_white_error=True, include_kernel_error=True, kernel_error=None, residuals_with_noise=None):
        """Computes the errors added in quadrature for all datasets corresponding to this kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The data errors.
        """
        
        # Get intrinsic data errors
        errors = self.get_intrinsic_data_errors()
        
        # Square
        errors = errors**2
        
        # Add per-instrument jitter terms in quadrature
        if include_white_error:
            for data in self.data.values():
                inds = self.data_inds[data.label]
                pname = "jitter_" + data.label
                errors[inds] += pars[pname].value**2
            
        # Compute GP error
        if include_kernel_error:
            for data in self.data.values():
                inds = self.data_inds[data.label]
                if kernel_error is None:
                    _, _kernel_error = self.realize(pars, residuals_with_noise=residuals_with_noise, xpred=data.t, wavelength=data.wavelength, return_kernel_error=True)
                errors[inds] += _kernel_error**2
                    
        # Square root
        errors = errors**0.5

        return errors
    
    def compute_dist_matrix(self, x1=None, x2=None, wave1=None, wave2=None):
        """Computes the distance matrix. The distance matrix is 

        Args:
            x1 (np.ndarray): [description]
            x2 (np.ndarray): [description]
            instname (str, optional): The name of the instrument this is for. Defaults to None.
        """
        if x1 is None:
            x1 = self.t
        if x2 is None:
            x2 = self.t
        super().compute_dist_matrix(x1=x1, x2=x2)
        self.wave1 = wave1
        self.wave2 = wave2
        self.wave_vec = self.data.get_wave_vec()
        self.compute_wave_matrix(wave1=wave1, wave2=wave2)
        
    def compute_wave_matrix(self, wave1=None, wave2=None):
        """Generates a matrix for a linear kernel.

        Args:
            wave1 (float, optional): [description]. Defaults to the data wavelengths.
            wave2 (float, optional): The wavelength for the second axis. Defaults to the data wavelengths.
        """
        n1, n2 = self.dist_matrix.shape
        if wave1 is not None:
            wave_vec1 = np.full(n1, fill_value=wave1)
        else:
            wave_vec1 = np.copy(self.wave_vec)
            
        if wave2 is not None:
            wave_vec2 = np.full(n2, fill_value=wave2)
        else:
            wave_vec2 = np.copy(self.wave_vec)
            
        # Compute matrices
        self.wave_diffs = np.zeros((n1, n2))
        self.freq_matrix = np.zeros((n1, n2))
        for i in range(n1):
             for j in range(n2):
                 self.wave_diffs[i, j] = np.abs(wave_vec1[i] - wave_vec2[j])
                 self.freq_matrix[i, j] = self.wavelength0 / np.sqrt(wave_vec1[i] * wave_vec2[j])
                    
        #self.wave_matrix = 1 / self.freq_matrix

    def make_wave_vec_unique(self):
        wave_vec = self.make_wave_vec()
        wave_vec_unique = np.unique(wave_vec)
        wave_vec_unique = np.sort(wave_vec_unique)
        return wave_vec_unique
        
    def realize(self, pars, residuals_with_noise, xpred=None, xres=None, return_kernel_error=False, wavelength=None):
        """Realize the GP (predict/ssample at arbitrary points). Meant to be the same as the predict method offered by other codes.

        Args:
            pars (Parameters): The parameters to use.
            residuals (np.ndarray): The residuals before the GP is subtracted.
            xpred (np.ndarray): The vector to realize the GP on.
            xres (np.ndarray): The vector the data is on.
            errors (np.ndarray): The errorbars, already added in quadrature.
            wavelength (float): The wavelength to realize the GP for.
            return_unc (bool, optional): Whether or not to compute the uncertainty in the GP. If True, both the mean and stddev are returned in a tuple. Defaults to False.

        Returns:
            np.ndarray OR tuple: If stddev is False, only the mean GP is returned. If stddev is True, the uncertainty in the GP is computed and returned as well. The mean GP is computed through a linear optimization.
        """
        
        # Resolve grids
        if xres is None:
            xres = self.data.get_vec('t')
        if xpred is None:
            xpred = xres
        
        # Get K
        self.compute_dist_matrix(xres, xres, wave1=None, wave2=None)
        K = self.compute_cov_matrix(pars, include_white_error=True)
        
        # Compute version of K without errorbars
        self.compute_dist_matrix(xpred, xres, wave1=wavelength, wave2=None)
        Ks = self.compute_cov_matrix(pars, include_white_error=False)

        # Avoid overflow errors in det(K) by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, residuals_with_noise)
        mu = np.dot(Ks, alpha).flatten()

        # Compute the uncertainty in the GP fitting.
        if return_kernel_error:
            self.compute_dist_matrix(xpred, xpred, wave1=wavelength, wave2=wavelength)
            Kss = self.compute_cov_matrix(pars, include_white_error=False)
            B = cho_solve(L, Ks.T)
            var = np.array(np.diag(Kss - np.dot(Ks, B))).flatten()
            unc = np.sqrt(var)
            self.compute_dist_matrix(wave1=None, wave2=None)
            return mu, unc
        else:
            self.compute_dist_matrix(wave1=None, wave2=None)
            return mu
        
    def make_wave_vec(self):
        wave_vec = np.array([], dtype=float)
        t = self.data.get_vec('t', sort=False)
        ss = np.argsort(t)
        for data in self.data.values():
            wave_vec = np.concatenate((wave_vec, np.full(data.t.size, fill_value=data.wavelength)))
        wave_vec = wave_vec[ss]
        return wave_vec
    
    def get_wave_inds(self, wavelength):
        inds = np.where(self.wave_vec == wavelength)[0]
        return inds
    
    def get_instnames_for_wave(self, wavelength):
        instnames = []
        for data in self.data.values():
            if data.wavelength == wavelength:
                instnames.append(data.label)
        return instnames
    
    @property
    def t(self):
        return self.x
    
    
class RVColor2(RVColor):
    
    def __init__(self, data, par_names=None):
        optkernels.GaussianProcess.__init__(self, data=data, par_names=par_names)
        self.tel_vec = self.data.make_tel_vec()
        self.compute_dist_matrix()
        self.n_instruments = len(self.data)
    
    def compute_cov_matrix(self, pars, include_white_error=True):
        
        # Alias params
        amp_matrix = self.make_amp_matrix(pars)
        eta2 = pars[self.par_names[self.n_instruments]].value # decay
        eta3 = pars[self.par_names[self.n_instruments + 1]].value # period
        eta4 = pars[self.par_names[self.n_instruments + 2]].value # smoothing factor
            
        # Construct individual kernels
        decay_kernel = np.exp(-0.5 * (self.dist_matrix / eta2)**2)
        periodic_kernel = np.exp(-0.5 * (1 / eta4)**2 * np.sin(np.pi * self.dist_matrix / eta3)**2)
        
        # Construct full cov matrix
        cov_matrix = amp_matrix * decay_kernel * periodic_kernel
        
        # Apply intrinsic plus white noise data errors
        if include_white_error:
            data_errors = self.compute_data_errors(pars, include_white_error=include_white_error, include_kernel_error=False)
            np.fill_diagonal(cov_matrix, np.diag(cov_matrix) + data_errors**2)
        
        return cov_matrix
    
    def compute_dist_matrix(self, x1=None, x2=None, instname1=None, instname2=None):
        """Computes the distance matrix. The distance matrix is 

        Args:
            x1 (np.ndarray): The vector for dimension 1.
            x2 (np.ndarray): The vector for dimension 2.
            instname1 (str, optional): The name of the instrument for dimension 1. Defaults to None, using the actual data.
            instname2 (str, optional): The name of the instrument for dimension 2. Defaults to None, using the actual data.
        """
        if x1 is None:
            x1 = self.t
        if x2 is None:
            x2 = self.t
        self.instname1 = instname1
        self.instname2 = instname2
        optkernels.CorrelatedNoiseKernel.compute_dist_matrix(self, x1=x1, x2=x2)
    
    def make_amp_matrix(self, pars):
        n1, n2 = self.dist_matrix.shape
        amp_vec1 = np.zeros(n1)
        amp_vec2 = np.zeros(n2)
        if self.instname1 is not None:
            amp_vec1[:] = pars["gp_amp_" + self.instname1].value
        else:
            for i in range(n1):
                amp_vec1[i] = pars["gp_amp_" + self.tel_vec[i]].value
        if self.instname2 is not None:
            amp_vec2[:] = pars["gp_amp_" + self.instname2].value
        else:
            for i in range(n2):
                amp_vec2[i] = pars["gp_amp_" + self.tel_vec[i]].value
        amp_matrix = np.outer(amp_vec1, amp_vec2)
        return amp_matrix
    
    def compute_data_errors(self, pars, include_white_error=True, include_kernel_error=False, kernel_error=None, residuals_with_noise=None):
        """Computes the errors added in quadrature for all datasets corresponding to this kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The data errors.
        """
        
        # Get intrinsic data errors
        errors = self.get_intrinsic_data_errors()
        
        # Square
        errors = errors**2
        
        # Add per-instrument jitter terms in quadrature
        if include_white_error:
            for data in self.data.values():
                inds = self.data_inds[data.label]
                pname = "jitter_" + data.label
                errors[inds] += pars[pname].value**2
            
        # Compute GP error
        if include_kernel_error:
            for data in self.data.values():
                inds = self.data_inds[data.label]
                if kernel_error is None:
                    _, _kernel_error = self.realize(pars, residuals_with_noise=residuals_with_noise, xpred=data.t, instrument=data.label, return_kernel_error=True)
                errors[inds] += _kernel_error**2
                    
        # Square root
        errors = errors**0.5

        return errors
    
    def realize(self, pars, residuals_with_noise, xpred=None, xres=None, return_kernel_error=False, instrument=None):
        """Realize the GP (predict/ssample at arbitrary points). Meant to be the same as the predict method offered by other codes.

        Args:
            pars (Parameters): The parameters to use.
            residuals (np.ndarray): The residuals before the GP is subtracted.
            xpred (np.ndarray): The vector to realize the GP on.
            xres (np.ndarray): The vector the data is on.
            errors (np.ndarray): The errorbars, already added in quadrature.
            instrument (float): The instrument to realize the GP for.
            return_unc (bool, optional): Whether or not to compute the uncertainty in the GP. If True, both the mean and stddev are returned in a tuple. Defaults to False.

        Returns:
            np.ndarray OR tuple: If stddev is False, only the mean GP is returned. If stddev is True, the uncertainty in the GP is computed and returned as well. The mean GP is computed through a linear optimization.
        """
        
        # Resolve grids
        if xres is None:
            xres = self.data.get_vec('t')
        if xpred is None:
            xpred = xres
        
        # Get K
        self.compute_dist_matrix(xres, xres, instname1=None, instname2=None)
        K = self.compute_cov_matrix(pars, include_white_error=True)
        
        # Compute version of K without errorbars
        self.compute_dist_matrix(xpred, xres, instname1=instrument, instname2=None)
        Ks = self.compute_cov_matrix(pars, include_white_error=False)

        # Avoid overflow errors in det(K) by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, residuals_with_noise)
        mu = np.dot(Ks, alpha).flatten()

        # Compute the uncertainty in the GP fitting.
        if return_kernel_error:
            self.compute_dist_matrix(xpred, xpred, instname1=instrument, instname2=instrument)
            Kss = self.compute_cov_matrix(pars, include_white_error=False)
            B = cho_solve(L, Ks.T)
            var = np.array(np.diag(Kss - np.dot(Ks, B))).flatten()
            unc = np.sqrt(var)
            self.compute_dist_matrix(xres, xres, instname1=None, instname2=None) # Reset
            return mu, unc
        else:
            self.compute_dist_matrix(xres, xres, instname1=None, instname2=None) # Reset
            return mu
        
        
class RVColor3(RVColor):
    
    # Alias params
    def compute_cov_matrix(self, pars, include_white_error=True):
        
        eta1 = pars[self.par_names[0]].value # amp linear wave scale
        eta2 = pars[self.par_names[1]].value # amp power law wave scale
        eta3 = pars[self.par_names[2]].value # decay
        eta4 = pars[self.par_names[3]].value # period
        eta5 = pars[self.par_names[4]].value # smoothing factor
        eta6 = pars[self.par_names[5]].value # decorrelation factor to scale with difference in frequency (f2 - f1)
            
        # Construct individual kernels
        lin_kernel = (eta1 * self.freq_matrix**eta2)**2
        decay_kernel = np.exp(-0.5 * (self.dist_matrix / eta3)**2)
        periodic_kernel = np.exp(-0.5 * (1 / eta5)**2 * np.sin(np.pi * self.dist_matrix / eta4)**2)
        decorr_kernel = np.exp(-0.5 * (self.freq_diffs / eta6)**2)
        
        # Construct full cov matrix
        cov_matrix = lin_kernel * decay_kernel * periodic_kernel * decorr_kernel
        
        # Apply intrinsic plus white noise data errors
        if include_white_error:
            data_errors = self.compute_data_errors(pars, include_white_error=include_white_error, include_kernel_error=False)
            np.fill_diagonal(cov_matrix, np.diag(cov_matrix) + data_errors**2)
        
        return cov_matrix
    
    def compute_wave_matrix(self, wave1=None, wave2=None):
        """Generates a matrix for a linear kernel.

        Args:
            wave1 (float, optional): [description]. Defaults to the data wavelengths.
            wave2 (float, optional): The wavelength for the second axis. Defaults to the data wavelengths.
        """
        n1, n2 = self.dist_matrix.shape
        if wave1 is not None:
            wave_vec1 = np.full(n1, fill_value=wave1)
        else:
            wave_vec1 = np.copy(self.wave_vec)
            
        if wave2 is not None:
            wave_vec2 = np.full(n2, fill_value=wave2)
        else:
            wave_vec2 = np.copy(self.wave_vec)
            
        # Compute matrices
        self.wave_diffs = np.zeros((n1, n2))
        self.freq_matrix = np.zeros((n1, n2))
        self.freq_diffs = np.zeros((n1, n2))
        for i in range(n1):
             for j in range(n2):
                 self.wave_diffs[i, j] = np.abs(wave_vec1[i] - wave_vec2[j])
                 self.freq_diffs[i, j] = np.abs(1 / wave_vec1[i] - 1 / wave_vec2[j])
                 self.freq_matrix[i, j] = self.wavelength0 / np.sqrt(wave_vec1[i] * wave_vec2[j])
                    
        #self.wave_matrix = 1 / self.freq_matrix