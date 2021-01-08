import numpy as np
from scipy.linalg import cho_solve, cho_factor
import matplotlib.pyplot as plt
import optimize.kernels as optkernels
from numba import njit, prange

class RVColor(optkernels.GaussianProcess):
    
    is_diag = False
    
    def __init__(self, data, par_names=None, wavelength0=550):
        self.wavelength0 = wavelength0
        super().__init__(data=data, par_names=par_names)
        self.n_data_points = len(self.data.get_vec('t'))
        self.wave_vec = self.make_wave_vec()
        self.unique_wavelengths = self.make_wave_vec_unique()
        self.compute_dist_matrix()
        
    @property
    def t(self):
        return self.x
    
    def compute_cov_matrix(self, pars, apply_errors=True):
        
        # Alias params
        eta1 = pars[self.par_names[0]].value # amp linear wave scale
        eta2 = pars[self.par_names[1]].value # amp power law wave scale
        eta3 = pars[self.par_names[2]].value # decay
        eta4 = pars[self.par_names[3]].value # period
        eta5 = pars[self.par_names[4]].value # smoothing factor
        
        # Data errors
        if apply_errors:
            data_errors = self.compute_data_errors(pars)
        else:
            data_errors = None
            
        lin_kernel = (eta1 * self.wave_matrix)**(2 * eta2)
        decay_kernel = np.exp(-0.5 * (self.dist_matrix / eta3)**2)
        periodic_kernel = np.exp(-0.5 * np.sin(np.pi * self.dist_matrix / eta4)**2)
        
        # Construct full cov matrix
        cov_matrix = lin_kernel * decay_kernel * periodic_kernel
        
        # Apply data errors
        if apply_errors:
            np.fill_diagonal(cov_matrix, np.diag(cov_matrix) + data_errors**2)
        
        return cov_matrix
                    
    def compute_data_errors(self, pars):
        """Computes the errors added in quadrature for all datasets corresponding to this kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The errors
        """
        errors = self.get_data_errors()
        errors **= 2
        for label in self.data:
            errors[self.data_inds[label]] += pars['jitter_' + label].value**2
        errors **= 0.5

        return errors
    
    def compute_dist_matrix(self, x1=None, x2=None, wave1=None, wave2=None):
        """Computes the distance matrix.

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
        self.compute_wave_matrix(wave1=wave1, wave2=wave2)
        
    def compute_wave_matrix(self, wave1=None, wave2=None):
        """Generates a matrix for a linear kernel.

        Args:
            wave1 (float, optional): [description]. Defaults to the data wavelengths.
            wave2 (float, optional): The wavelength for the second axis. Defaults to the data wavelengths.
        """
        if wave1 is not None:
            wave_vec1 = np.full(self.dist_matrix.shape[0], fill_value=wave1)
        else:
            wave_vec1 = self.wave_vec
            
        if wave2 is not None:
            wave_vec2 = np.full(self.dist_matrix.shape[1], fill_value=wave2)
        else:
            wave_vec2 = self.wave_vec
            
        # Compute matrix
        self.wave_matrix = self.wavelength0 / np.outer(wave_vec1, wave_vec2)**0.5

    def make_wave_vec_unique(self):
        wave_vec = self.make_wave_vec()
        wave_vec_unique = np.unique(wave_vec)
        wave_vec_unique = np.sort(wave_vec_unique)
        return wave_vec_unique
        
    def realize(self, pars, residuals, xpred=None, xres=None, return_unc=False, wavelength=None):
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
        K = self.compute_cov_matrix(pars, apply_errors=True)
        
        # Compute version of K without errorbars
        self.compute_dist_matrix(xpred, xres, wave1=wavelength, wave2=None)
        Ks = self.compute_cov_matrix(pars, apply_errors=False)

        # Avoid overflow errors in det(K) by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, residuals)
        mu = np.dot(Ks, alpha).flatten()

        # Compute the uncertainty in the GP fitting.
        if return_unc:
            self.compute_dist_matrix(xpred, xpred, wave1=wavelength, wave2=wavelength)
            Kss = self.compute_cov_matrix(pars, apply_errors=False)
            B = cho_solve(L, Ks.T)
            var = np.array(np.diag(Kss - np.dot(Ks, B))).flatten()
            unc = np.sqrt(var)
            self.compute_dist_matrix(xres, xres, wave1=None, wave2=None)
            return mu, unc
        else:
            self.compute_dist_matrix(xres, xres, wave1=None, wave2=None)
            return mu
        
    def make_vec_for_wavelength(self, wavelength, kind='t'):
        vec = np.array([], dtype=float)
        t = np.array([], dtype=float)
        for data in self.data:
            if data.wavelength == wavelength:
                t = np.concatenate((t, data.get_vec('t', sort=False)))
                vec = np.concatenate((vec, data.get_vec(kind, sort=False)))
        ss = np.argsort(t)
        vec = vec[ss]
        return vec
        
    def make_wave_vec(self):
        wave_vec = np.array([], dtype=float)
        x = self.data.get_vec('x', sort=False)
        ss = np.argsort(x)
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