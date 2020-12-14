import numpy as np
from scipy.linalg import cho_solve, cho_factor
import matplotlib.pyplot as plt
import optimize.kernels as optkernels

class RVColor(optkernels.GaussianProcess):
    
    def compute_cov_matrix(self, pars, apply_errors=True, instname=False):
        
        # Alias params
        amp = pars["gp_amp"].value
        exp_length = pars["gp_decay"].value
        per = pars["gp_per"].value
        per_length = pars["gp_per_length"].value
        exponent_power = pars["gp_amp_exp"].value
        
        # Compute exp decay term
        decay_term = -0.5 * self.dist_matrix**2 / exp_length**2
        
        # Compute periodic term
        periodic_term = -0.5 * np.sin((np.pi / per) * self.dist_matrix)**2 / per_length**2
        
        # Add, exponentiate, and include amplitude
        amp_matrix = amp * self.wave_matrix**exponent_power
        cov_matrix = self.wave_matrix**2 * np.exp(decay_term + periodic_term)
        
        # Include errors on the diagonal
        if apply_errors:
            _errors = self.compute_data_errors(pars)
            errors_quad = np.diag(cov_matrix) + _errors**2
            np.fill_diagonal(cov_matrix, errors_quad)
        
        return cov_matrix
    
    def compute_dist_matrix(self, x1, x2):
        """Computes the distance matrix.

        Args:
            x1 (np.ndarray): [description]
            x2 (np.ndarray): [description]
            instname (str, optional): The name of the instrument this is for. Defaults to None.
        """
        super().compute_dist_matrix(x1, x2)
        self.compute_wave_matrix(x1, x2)
        
    def compute_wave_matrix(self, x1, x2, instname):
        """Computes the wavelength matrix, 

        Args:
            x1 (np.ndarray): The array of times for axis 1.
            x2 (np.ndarray): The array of times for axis 2.
            wavelength ([type], optional): [description]. Defaults to None.
        """
        if wavelength is None: 
            assert x1.size == x2.size
            wavelengths = {self.data.label : data.label.wavelength for data in self.data}
            wave_vector = np.array([wavelengths[inst] for inst in self.data.telvec], dtype=float)
            self.wave_matrix = self.wavelength0 / np.sqrt(np.outer(wave_vector, wave_vector))
        else:
            self.wave_matrix = np.full(shape=(x1.size, x2.size), fill_value=self.wavelength0 / (wavelength * wavelength)**0.5)
            
            
    def realize(self, pars, xpred, xres, res, instname=None, errors=None, stddev=False):
        """Realize the GP (sample at arbitrary points). Meant to be the same as the predict method offered by other codes.

        Args:
            pars (Parameters): The parameters to use.
            xpred (np.ndarray): The vector to realize the GP on.
            xres (np.ndarray): The vector the data is on.
            res (np.ndarray): The residuals before the GP is subtracted.
            errors (np.ndarray): The instrinsic errorbars.
            instname (np.ndarray): The instrument to realize the GP for.
            stddev (bool, optional): Whether or not to compute the uncertainty in the GP. If True, both the mean and stddev are returned in a tuple. Defaults to False.

        Returns:
            np.ndarray OR tuple: If stddev is False, only the mean GP is returned. If stddev is True, the uncertainty in the GP is computed and returned as well. The mean GP is realized through a linear optimization.
        """
        
        # Get K
        self.compute_dist_matrix(xres, xres)
        K = self.compute_cov_matrix(pars, errors=errors)
        
        # Compute version of K without errorbars
        self.compute_dist_matrix(xpred, xres)
        Ks = self.compute_cov_matrix(pars, errors=None)

        # Avoid overflow errors in det(K) by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, res)
        mu = np.dot(Ks, alpha).flatten()

        if stddev:
            self.compute_dist_matrix(xpred, xpred)
            Kss = self.compute_cov_matrix(pars, errors=None)
            B = cho_solve(L, Ks.T)
            var = np.array(np.diag(Kss - np.dot(Ks, B))).flatten()
            unc = np.sqrt(var)
            return mu, stddev
        else:
            return mu