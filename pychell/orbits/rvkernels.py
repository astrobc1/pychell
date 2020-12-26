import numpy as np
from scipy.linalg import cho_solve, cho_factor
import matplotlib.pyplot as plt
import optimize.kernels as optkernels

class RVColor(optkernels.GaussianProcess):
    
    is_diag = False
    
    def __init__(self, *args, wavelength0=550, **kwargs):
        self.wavelength0 = wavelength0
        super().__init__(*args, **kwargs)
        
    @property
    def t(self):
        return self.x
    
    def compute_cov_matrix(self, pars, apply_errors=True):
        
        # Alias params
        amp = pars[self.par_names[0]].value
        wave_scale = pars[self.par_names[1]].value
        exp_length = pars[self.par_names[2]].value
        per = pars[self.par_names[3]].value
        per_length = pars[self.par_names[4]].value
        
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
    
    def compute_data_errors(self, pars):
        """Computes the errors added in quadrature for all datasets corresponding to this kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The errors
        """
        if self.current_instname is not None:
            errors = np.copy(self.data[self.current_instname].rverr)
            errors **= 2
            errors += pars['jitter_' + self.current_instname].value**2
        else:
            errors = self.get_data_errors()
            errors **= 2
            for label in self.data:
                errors[self.data_inds[label]] += pars['jitter_' + label].value**2
        errors **= 0.5
        return errors
    
    def compute_dist_matrix(self, x1=None, x2=None, instname=None):
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
        self.current_instname = instname
        super().compute_dist_matrix(x1=x1, x2=x2)
        self.compute_wave_matrix(x1, x2, instname=instname)
        
    def compute_wave_matrix(self, x1, x2, instname=None):
        """Computes the wavelength matrix, sqrt(lambda_1 * lambda_2)

        Args:
            x1 (np.ndarray): The array of times for axis 1.
            x2 (np.ndarray): The array of times for axis 2.
            wavelength (float, optional): The wavelength of the instrument to use. If None (default), the full wavelength matrix is computed.
        """
        if instname is None:
            assert x1.size == x2.size
            wave_vector = self.make_wave_vec()
            self.wave_matrix = self.wavelength0 / np.sqrt(np.outer(wave_vector, wave_vector))
        else:
            w = self.data[instname].wavelength
            self.wave_matrix = np.full(shape=(x1.size, x2.size), fill_value=self.wavelength0 / (w * w)**0.5)
            
    def realize(self, pars, residuals, xpred=None, xres=None, return_unc=False, instname=None, **kwargs):
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
        
        # Resolve grids
        if xres is None:
            if instname is None:
                xres = self.data.get_vec('x')
            else:
                xres = self.data[instname].t
        if xpred is None:
            xpred = xres
        
        # Get K
        self.compute_dist_matrix(xres, xres, instname=instname)
        K = self.compute_cov_matrix(pars, apply_errors=True)
        
        # Compute version of K without errorbars
        self.compute_dist_matrix(xpred, xres, instname=instname)
        Ks = self.compute_cov_matrix(pars, apply_errors=False)

        # Avoid overflow errors in det(K) by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, residuals)
        mu = np.dot(Ks, alpha).flatten()

        # Compute the uncertainty in the GP fitting.
        if return_unc:
            self.compute_dist_matrix(xpred, xpred, instname=instname)
            Kss = self.compute_cov_matrix(pars, apply_errors=False)
            B = cho_solve(L, Ks.T)
            var = np.array(np.diag(Kss - np.dot(Ks, B))).flatten()
            unc = np.sqrt(var)
            self.compute_dist_matrix()
            return mu, unc
        else:
            self.compute_dist_matrix()
            return mu
        
    def make_wave_vec(self):
        wave_vec = np.array([], dtype=float)
        x = self.data.get_vec('x', sort=False)
        ss = np.argsort(x)
        for data in self.data.values():
            wave_vec = np.concatenate((wave_vec, np.full(data.t.size, fill_value=data.wavelength)))
        wave_vec = wave_vec[ss]
        return wave_vec