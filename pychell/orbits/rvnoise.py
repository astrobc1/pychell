import numpy as np
from scipy.linalg import cho_solve, cho_factor
import matplotlib.pyplot as plt
import optimize.noise as optnoise
from numba import njit, prange

# Import optimize kernels into namespace
from optimize.noise import *

class ChromaticKernelJ1(optnoise.NoiseKernel):
    
    def __init__(self, data, par_names):
        super().__init__(data=data)
        self.par_names = par_names
        self.instname_vec = self.data.gen_instname_vec()
    
    def compute_cov_matrix(self, pars):
        
        # Number of instruments
        n_instruments = len(self.data)
        
        # Alias params
        amp_matrix = self.gen_amp_matrix(pars)
        eta2 = pars[self.par_names[n_instruments]].value # decay
        eta3 = pars[self.par_names[n_instruments + 1]].value # period
        eta4 = pars[self.par_names[n_instruments + 2]].value # smoothing factor

        # Construct QP terms
        decay_term = -0.5 * (self.dist_matrix / eta2)**2
        periodic_term = -0.5 * (1 / eta4)**2 * np.sin(np.pi * self.dist_matrix / eta3)**2
        
        # Construct full cov matrix
        K = amp_matrix * np.exp(decay_term + periodic_term)
        
        return K
    
    def gen_amp_matrix(self, pars):
        
        # The current shape of the covariance matrix
        n1, n2 = self.dist_matrix.shape
        
        # Initialize amp vectors
        amp_vec1 = np.zeros(n1)
        amp_vec2 = np.zeros(n2)
        
        # Fill each
        if self.instname1 is not None:
            amp_vec1[:] = pars[f"gp_amp_{self.instname1}"].value
        else:
            for i in range(n1):
                amp_vec1[i] = pars[f"gp_amp_{self.instname_vec[i]}"].value
        if self.instname2 is not None:
            amp_vec2[:] = pars[f"gp_amp_{self.instname2}"].value
        else:
            for i in range(n2):
                amp_vec2[i] = pars[f"gp_amp_{self.instname_vec[i]}"].value
                
        # Outer product: A_ij = a1 * a2 and A_ij = A_ji
        A = np.outer(amp_vec1, amp_vec2)
        
        return A
    
    def initialize(self, x1=None, x2=None, instname1=None, instname2=None):
        super().initialize(x1=x1, x2=x2)
        self.instname1 = instname1
        self.instname2 = instname2


class ChromaticProcessJ1(optnoise.GaussianProcess):
    
    def __init__(self, data, par_names):
        super().__init__(data=data, kernel=ChromaticKernelJ1(data=data, par_names=par_names))
        self.initialize()
        
    def compute_cov_matrix(self, pars, include_uncorr_error=True):
        
        # Compute kernel
        K = self.kernel.compute_cov_matrix(pars)
        
        # Uncorrelated errors (intrinsic error bars and additional per-data label jitter)
        if include_uncorr_error:
            assert K.shape[0] ==  K.shape[1]
            data_errors = self.compute_data_errors(pars)
            assert K.shape[0] == data_errors.size
            np.fill_diagonal(K, np.diagonal(K) + data_errors**2)
        
        return K
                    
    def compute_data_errors(self, pars, include_gp_error=False, gp_error=None, data_with_noise=None):
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
        for data in self.data.values():
            inds = self.data.indices[data.label]
            pname = "jitter_" + data.label
            errors[inds] += pars[pname].value**2
            
        # Compute GP error
        if include_gp_error:
            for data in self.data.values():
                inds = self.data.indices[data.label]
                if gp_error is None:
                    _, _gp_error = self.realize(pars, data_with_noise=data_with_noise, xpred=data.t, return_gp_error=True, instname=data.label)
                errors[inds] += _gp_error**2
                    
        # Square root
        errors = errors**0.5

        return errors
        
    def realize(self, pars, data_with_noise, instname, xpred=None, xdata=None, return_gp_error=False):
        """Realize the GP (predict/ssample at arbitrary points).

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
        
        # Resolve the grids to use.
        if xdata is None:
            xdata = self.data.gen_vec("x")
        if xpred is None:
            xpred = xdata
        
        # Get K
        self.initialize(x1=xdata, x2=xdata)
        K = self.compute_cov_matrix(pars, include_uncorr_error=True)
        
        # Compute version of K without errorbars
        self.initialize(x1=xpred, x2=xdata, instname1=instname, instname2=None)
        Ks = self.compute_cov_matrix(pars, include_uncorr_error=False)

        # Avoid overflow errors by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, data_with_noise)
        mu = np.dot(Ks, alpha).flatten()

        # Compute the uncertainty in the GP fitting.
        if return_gp_error:
            self.initialize(x1=xpred, x2=xpred, instname1=instname, instname2=instname)
            Kss = self.compute_cov_matrix(pars, include_uncorr_error=False)
            B = cho_solve(L, Ks.T)
            var = np.array(np.diag(Kss - np.dot(Ks, B))).flatten()
            unc = np.sqrt(var)
            self.initialize()
            return mu, unc
        else:
            self.initialize()
            return mu
    
    def initialize(self, x1=None, x2=None, instname1=None, instname2=None):
        self.kernel.initialize(x1=x1, x2=x2, instname1=instname1, instname2=instname2)
    
    
class ChromaticKernelJ2(optnoise.NoiseKernel):
    
    def __init__(self, data, par_names, wavelength0=550):
        super().__init__(data=data)
        self.par_names = par_names
        self.wavelength0 = wavelength0
        self.instname_vec = self.data.gen_instname_vec()
        self.wave_vec = self.data.gen_wave_vec()
    
    def compute_cov_matrix(self, pars):
        
        # Alias params
        eta1 = pars[self.par_names[0]].value # amp linear wave scale
        eta2 = pars[self.par_names[1]].value # amp power law wave scale
        eta3 = pars[self.par_names[2]].value # decay
        eta4 = pars[self.par_names[3]].value # period
        eta5 = pars[self.par_names[4]].value # smoothing factor
            
        # Construct individual kernels
        lin_kernel = (eta1 * self.freq_matrix**eta2)**2
        decay_term = -0.5 * (self.dist_matrix / eta3)**2
        periodic_term = -0.5 * (1 / eta5)**2 * np.sin(np.pi * self.dist_matrix / eta4)**2
        
        # Construct full cov matrix
        K = lin_kernel * np.exp(decay_term + periodic_term)
        
        return K
    
    def gen_wave_matrix(self, wave1=None, wave2=None):
        
        # The current shape of the covariance matrix
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
    
    def initialize(self, x1=None, x2=None, wave1=None, wave2=None):
        super().initialize(x1=x1, x2=x2)
        self.gen_wave_matrix(wave1=wave1, wave2=wave2)
        
    @property
    def unique_wavelengths(self):
        return np.sort(np.unique(self.wave_vec))


class ChromaticProcessJ2(optnoise.GaussianProcess):
    
    def __init__(self, data, par_names):
        super().__init__(data=data, kernel=ChromaticKernelJ2(data=data, par_names=par_names))
        self.initialize()
        
    def compute_cov_matrix(self, pars, include_uncorr_error=True):
        
        # Compute kernel
        K = self.kernel.compute_cov_matrix(pars)
        
        # Uncorrelated errors (intrinsic error bars and additional per-data label jitter)
        if include_uncorr_error:
            assert K.shape[0] ==  K.shape[1]
            data_errors = self.compute_data_errors(pars)
            assert K.shape[0] == data_errors.size
            np.fill_diagonal(K, np.diagonal(K) + data_errors**2)
        
        return K
                    
    def compute_data_errors(self, pars, include_gp_error=False, gp_error=None, data_with_noise=None):
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
        for data in self.data.values():
            inds = self.data.indices[data.label]
            pname = "jitter_" + data.label
            errors[inds] += pars[pname].value**2
            
        # Compute GP error
        if include_gp_error:
            for data in self.data.values():
                inds = self.data.indices[data.label]
                if gp_error is None:
                    _, _gp_error = self.realize(pars, data_with_noise=data_with_noise, xpred=data.t, return_gp_error=True, wavelength=data.wavelength)
                errors[inds] += _gp_error**2
                    
        # Square root
        errors = errors**0.5

        return errors
        
    def realize(self, pars, data_with_noise, wavelength, xpred=None, xdata=None, return_gp_error=False):
        """Realize the GP (predict/ssample at arbitrary points).

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
        
        # Resolve the grids to use.
        if xdata is None:
            xdata = self.data.gen_vec("x")
        if xpred is None:
            xpred = xdata
        
        # Get K
        self.initialize(x1=xdata, x2=xdata)
        K = self.compute_cov_matrix(pars, include_uncorr_error=True)
        
        # Compute version of K without errorbars
        self.initialize(x1=xpred, x2=xdata, wave1=wavelength, wave2=None)
        Ks = self.compute_cov_matrix(pars, include_uncorr_error=False)

        # Avoid overflow errors by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, data_with_noise)
        mu = np.dot(Ks, alpha).flatten()

        # Compute the uncertainty in the GP fitting.
        if return_gp_error:
            self.initialize(x1=xpred, x2=xpred, wave1=wavelength, wave2=wavelength)
            Kss = self.compute_cov_matrix(pars, include_uncorr_error=False)
            B = cho_solve(L, Ks.T)
            var = np.array(np.diag(Kss - np.dot(Ks, B))).flatten()
            unc = np.sqrt(var)
            self.initialize()
            return mu, unc
        else:
            self.initialize()
            return mu
    
    def initialize(self, x1=None, x2=None, wave1=None, wave2=None):
        self.kernel.initialize(x1=x1, x2=x2, wave1=wave1, wave2=wave2)
    
    def get_wave_inds(self, wavelength):
        inds = np.where(self.kernel.wave_vec == wavelength)[0]
        return inds
    
    def get_instnames_for_wave(self, wavelength):
        instnames = []
        for data in self.data.values():
            if data.wavelength == wavelength:
                instnames.append(data.label)
        return instnames
        
    @property
    def unique_wavelengths(self):
        return np.sort(np.unique(self.kernel.wave_vec))


class ChromaticKernelJ3(optnoise.NoiseKernel):
    
    def __init__(self, data, par_names):
        super().__init__(data=data)
        self.par_names = par_names
        self.instname_vec = self.data.gen_instname_vec()
        self.wave_vec = self.data.gen_wave_vec()
    
    def compute_cov_matrix(self, pars):
        
        # Number of instruments
        n_instruments = len(self.data)
        
        # Alias params
        amp_matrix = self.gen_amp_matrix(pars)
        eta2 = pars[self.par_names[n_instruments]].value # decay
        eta3 = pars[self.par_names[n_instruments + 1]].value # period
        eta4 = pars[self.par_names[n_instruments + 2]].value # smoothing factor
        eta5 = pars[self.par_names[n_instruments + 3]].value # decorr param

        # Construct QP terms
        decay_term = -0.5 * (self.dist_matrix / eta2)**2
        periodic_term = -0.5 * (1 / eta4)**2 * np.sin(np.pi * self.dist_matrix / eta3)**2
        decorr_kernel = np.exp(-0.5 * (self.wave_diffs / eta5)**2)
        
        # Construct full cov matrix
        K = amp_matrix * np.exp(decay_term + periodic_term) * decorr_kernel
        
        return K
    
    def gen_amp_matrix(self, pars):
        
        # The current shape of the covariance matrix
        n1, n2 = self.dist_matrix.shape
        
        # Initialize amp vectors
        amp_vec1 = np.zeros(n1)
        amp_vec2 = np.zeros(n2)
        
        # Fill each
        if self.instname1 is not None:
            amp_vec1[:] = pars[f"gp_amp_{self.instname1}"].value
        else:
            for i in range(n1):
                amp_vec1[i] = pars[f"gp_amp_{self.instname_vec[i]}"].value
        if self.instname2 is not None:
            amp_vec2[:] = pars[f"gp_amp_{self.instname2}"].value
        else:
            for i in range(n2):
                amp_vec2[i] = pars[f"gp_amp_{self.instname_vec[i]}"].value
                
        # Outer product: A_ij = a1 * a2 and A_ij = A_ji
        A = np.outer(amp_vec1, amp_vec2)
        
        return A
    
    def initialize(self, x1=None, x2=None, instname1=None, instname2=None):
        super().initialize(x1=x1, x2=x2)
        wave1 = self.data[instname1].wavelength if instname1 is not None else None
        wave2 = self.data[instname2].wavelength if instname2 is not None else None
        self.gen_wave_matrix(wave1=wave1, wave2=wave2)
        self.instname1 = instname1
        self.instname2 = instname2
    
    def gen_wave_matrix(self, wave1=None, wave2=None):
        
        # The current shape of the covariance matrix
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
        self.wave_diffs = optnoise.compute_stationary_dist_matrix(wave_vec1, wave_vec2)


class ChromaticProcessJ3(optnoise.GaussianProcess):
    
    def __init__(self, data, par_names):
        super().__init__(data=data, kernel=ChromaticKernelJ3(data=data, par_names=par_names))
        self.initialize()
        
    def compute_cov_matrix(self, pars, include_uncorr_error=True):
        
        # Compute kernel
        K = self.kernel.compute_cov_matrix(pars)
        
        # Uncorrelated errors (intrinsic error bars and additional per-data label jitter)
        if include_uncorr_error:
            assert K.shape[0] ==  K.shape[1]
            data_errors = self.compute_data_errors(pars)
            assert K.shape[0] == data_errors.size
            np.fill_diagonal(K, np.diagonal(K) + data_errors**2)
        
        return K

    def compute_data_errors(self, pars, include_gp_error=False, gp_error=None, data_with_noise=None):
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
        for data in self.data.values():
            inds = self.data.indices[data.label]
            pname = "jitter_" + data.label
            errors[inds] += pars[pname].value**2
            
        # Compute GP error
        if include_gp_error:
            for data in self.data.values():
                inds = self.data.indices[data.label]
                if gp_error is None:
                    _, _gp_error = self.realize(pars, data_with_noise=data_with_noise, xpred=data.t, return_gp_error=True, instname=data.label)
                errors[inds] += _gp_error**2
                    
        # Square root
        errors = errors**0.5

        return errors

    def realize(self, pars, data_with_noise, instname, xpred=None, xdata=None, return_gp_error=False):
        """Realize the GP (predict/ssample at arbitrary points).

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
        
        # Resolve the grids to use.
        if xdata is None:
            xdata = self.data.gen_vec("x")
        if xpred is None:
            xpred = xdata
        
        # Get K
        self.initialize(x1=xdata, x2=xdata)
        K = self.compute_cov_matrix(pars, include_uncorr_error=True)
        
        # Compute version of K without errorbars
        self.initialize(x1=xpred, x2=xdata, instname1=instname, instname2=None)
        Ks = self.compute_cov_matrix(pars, include_uncorr_error=False)

        # Avoid overflow errors by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, data_with_noise)
        mu = np.dot(Ks, alpha).flatten()

        # Compute the uncertainty in the GP fitting.
        if return_gp_error:
            self.initialize(x1=xpred, x2=xpred, instname1=instname, instname2=instname)
            Kss = self.compute_cov_matrix(pars, include_uncorr_error=False)
            B = cho_solve(L, Ks.T)
            var = np.array(np.diag(Kss - np.dot(Ks, B))).flatten()
            unc = np.sqrt(var)
            self.initialize()
            return mu, unc
        else:
            self.initialize()
            return mu

    def initialize(self, x1=None, x2=None, instname1=None, instname2=None):
        self.kernel.initialize(x1=x1, x2=x2, instname1=instname1, instname2=instname2)
        


class ChromaticProcessJ4(optnoise.GaussianProcess):
    
    def __init__(self, data, par_names):
        super().__init__(data=data, kernel=ChromaticKernelJ1(data=data, par_names=par_names))
        self.initialize()
        
    def compute_cov_matrix(self, pars, include_uncorr_error=True):
        
        # Compute kernel
        K = self.kernel.compute_cov_matrix(pars)
        
        # Uncorrelated errors (intrinsic error bars and additional per-data label jitter)
        if include_uncorr_error:
            assert K.shape[0] ==  K.shape[1]
            data_errors = self.compute_data_errors(pars)
            assert K.shape[0] == data_errors.size
            np.fill_diagonal(K, np.diagonal(K) + data_errors**2)
        
        return K
                    
    def compute_data_errors(self, pars, include_gp_error=False, gp_error=None, data_with_noise=None):
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
        for data in self.data.values():
            inds = self.data.indices[data.label]
            pname = "jitter_" + data.label
            errors[inds] += pars[pname].value**2
            
        # Compute GP error
        if include_gp_error:
            for data in self.data.values():
                inds = self.data.indices[data.label]
                if gp_error is None:
                    _, _gp_error = self.realize(pars, data_with_noise=data_with_noise, xpred=data.t, return_gp_error=True, instname=data.label)
                errors[inds] += _gp_error**2
                    
        # Square root
        errors = errors**0.5

        return errors
        
    def realize(self, pars, data_with_noise, instname, xpred=None, xdata=None, return_gp_error=False):
        """Realize the GP (predict/ssample at arbitrary points).

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
        
        # Resolve the grids to use.
        if xdata is None:
            xdata = self.data.gen_vec("x")
        if xpred is None:
            xpred = xdata
        
        # Get K
        self.initialize(x1=xdata, x2=xdata)
        K = self.compute_cov_matrix(pars, include_uncorr_error=True)
        
        # Compute version of K without errorbars
        self.initialize(x1=xpred, x2=xdata, instname1=instname, instname2=None)
        Ks = self.compute_cov_matrix(pars, include_uncorr_error=False)

        # Avoid overflow errors by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, data_with_noise)
        mu = np.dot(Ks, alpha).flatten()

        # Compute the uncertainty in the GP fitting.
        if return_gp_error:
            self.initialize(x1=xpred, x2=xpred, instname1=instname, instname2=instname)
            Kss = self.compute_cov_matrix(pars, include_uncorr_error=False)
            B = cho_solve(L, Ks.T)
            var = np.array(np.diag(Kss - np.dot(Ks, B))).flatten()
            unc = np.sqrt(var)
            self.initialize()
            return mu, unc
        else:
            self.initialize()
            return mu
    
    def initialize(self, x1=None, x2=None, instname1=None, instname2=None):
        self.kernel.initialize(x1=x1, x2=x2, instname1=instname1, instname2=instname2)
    
