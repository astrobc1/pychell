import numpy as np
from scipy.linalg import cho_solve, cho_factor
import matplotlib.pyplot as plt
import optimize.kernels as optkernels

class RVColor(optkernels.GaussianProcess):
    
    def __init__(self, *args, wavelength0=550, **kwargs):
        self.wavelength0 = wavelength0
        super().__init__(*args, **kwargs)
        
    @property
    def t(self):
        return self.x
    
    def compute_cov_matrix(self, pars, apply_errors=True, instname=False):
        
        # Alias params
        amp = pars[self.par_names[0]].value
        exp_length = pars[self.par_names[0]].value
        per = pars[self.par_names[0]].value
        per_length = pars[self.par_names[0]].value
        wave_scale = pars[self.par_names[0]].value
        
        # Compute exp decay term
        decay_term = -0.5 * self.dist_matrix**2 / exp_length**2
        
        # Compute periodic term
        periodic_term = -0.5 * np.sin((np.pi / per) * self.dist_matrix)**2 / per_length**2
        
        # Add, exponentiate, and include amplitude
        amp_matrix = amp * self.wave_matrix**wave_scale
        cov_matrix = amp_matrix**2 * np.exp(decay_term + periodic_term)
        
        # Include errors on the diagonal
        if apply_errors:
            errors = self.compute_data_errors(pars)
            errors_quad = np.diag(cov_matrix) + errors**2
            np.fill_diagonal(cov_matrix, errors_quad)
        
        return cov_matrix
    
    def compute_dist_matrix(self, x1=None, x2=None, wavelength=None):
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
        self.compute_wave_matrix(x1, x2, wavelength=wavelength)
        
    def compute_wave_matrix(self, x1, x2, wavelength=None):
        """Computes the wavelength matrix, 

        Args:
            x1 (np.ndarray): The array of times for axis 1.
            x2 (np.ndarray): The array of times for axis 2.
            wavelength (float, optional): The wavelength of the instrument to use. If None (default), the full wavelength matrix is computed.
        """
        if wavelength is None:
            assert x1.size == x2.size
            wave_vector = self.data.make_wave_vec()
            self.wave_matrix = self.wavelength0 / np.sqrt(np.outer(wave_vector, wave_vector))
        else:
            self.wave_matrix = np.full(shape=(x1.size, x2.size), fill_value=self.wavelength0 / (wavelength * wavelength)**0.5)
            
    def realize(self, pars, residuals, xpred=None, xres=None, return_unc=False, wavelength=None, **kwargs):
        """Realize the GP (sample at arbitrary points). Meant to be the same as the predict method offered by other codes.

        Args:
            pars (Parameters): The parameters to use.
            residuals (np.ndarray): The residuals before the GP is subtracted.
            xpred (np.ndarray): The vector to realize the GP on.
            xres (np.ndarray): The vector the data is on.
            errors (np.ndarray): The errorbars, added in quadrature.
            return_unc (bool, optional): Whether or not to compute the uncertainty in the GP. If True, both the mean and stddev are returned in a tuple. Defaults to False.

        Returns:
            np.ndarray OR tuple: If stddev is False, only the mean GP is returned. If stddev is True, the uncertainty in the GP is computed and returned as well. The mean GP is computed through a linear optimization (i.e, minimiation surface is purely concave or convex).
        """
        
        # Resolve the grids to use.
        if xres is None:
            xres = self.data.get_vec('x')
        if xpred is None:
            xpred = xres
            
        # Resolve wavelength
        
        # Get K
        self.compute_dist_matrix(xres, xres)
        K = self.compute_cov_matrix(pars, apply_errors=True)
        
        # Compute version of K without errorbars
        self.compute_dist_matrix(xpred, xres)
        Ks = self.compute_cov_matrix(pars, apply_errors=False)

        # Avoid overflow errors in det(K) by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, residuals)
        mu = np.dot(Ks, alpha).flatten()

        # Compute the uncertainty in the GP fitting.
        if return_unc:
            self.compute_dist_matrix(xpred, xpred)
            Kss = self.compute_cov_matrix(pars, apply_errors=False)
            B = cho_solve(L, Ks.T)
            var = np.array(np.diag(Kss - np.dot(Ks, B))).flatten()
            unc = np.sqrt(var)
            return mu, unc
        else:
            return mu