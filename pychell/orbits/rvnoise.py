import numpy as np
from scipy.linalg import cho_solve, cho_factor
import matplotlib.pyplot as plt
import optimize.noise as optnoise
from numba import njit, prange

# Import optimize kernels into namespace
from optimize.noise import WhiteNoiseProcess, GaussianProcess, CorrelatedNoiseKernel

# Import rv kernels into namespace
from pychell.orbits.rvkernels import ChromaticKernelJ1

################
#### JITTER ####
################

class RVJitter(WhiteNoiseProcess):
    
    def compute_data_errors(self, pars):
        """Computes the errors added in quadrature for all datasets corresponding to this kernel.

        Args:
            pars (Parameters): The parameters to use.
            
        Returns:
            np.ndarray: The final data errors.
        """
    
        # Get intrinsic data errors
        errors = self.data.get_apriori_errors()
        
        # Add jitter in quadrature
        for label in self.data:
            inds = self.data.indices[label]
            errors[inds] = np.sqrt(errors[inds]**2 + pars[f"jitter_{label}"].value**2)
        
        return errors


#############
#### GPs ####
#############

class ChromaticProcessJ1(GaussianProcess):
    
    def __init__(self, data=None, name=None, par_names=None):
        super().__init__(data=data, name=name, kernel=ChromaticKernelJ1(data=data, par_names=par_names))
                    
    def compute_data_errors(self, pars, include_corr_error=False, linpred=None):
        """Computes the errors added in quadrature for all datasets corresponding to this kernel.
        Args:
            pars (Parameters): The parameters to use.
        Returns:
            np.ndarray: The data errors.
        """
        
        # Get intrinsic data errors
        errors = self.data.get_apriori_errors()
        
        # Add per-instrument jitter terms in quadrature
        for data in self.data.values():
            inds = self.data.indices[data.label]
            pname = f"jitter_{data.label}"
            errors[inds] = np.sqrt(errors[inds]**2 + pars[pname].value**2)
            
        # Compute GP error
        if include_corr_error:
            for data in self.data.values():
                inds = self.data.indices[data.label]
                gp_error = self.compute_corr_error(pars, linpred=linpred, xpred=data.t, instname=data.label)
                errors[inds] = np.sqrt(errors[inds]**2 + gp_error**2)
                    
        return errors
        
    def realize(self, pars, linpred, instname, xpred=None):
        """Realize the GP (predict/ssample at arbitrary points).
        Args:
            pars (Parameters): The parameters to use.
            residuals (np.ndarray): The residuals before the GP is subtracted.
            xpred (np.ndarray): The vector to realize the GP on.
            errors (np.ndarray): The errorbars, already added in quadrature.
        Returns:
            np.ndarray OR tuple: If stddev is False, only the mean GP is returned. If stddev is True, the uncertainty in the GP is computed and returned as well. The mean GP is computed through a linear optimization.
        """
        
        # Get grids
        xdata = self.data.x
        if xpred is None:
            xpred = xdata
        
        # Get K
        self.initialize(p0=pars, x1=xdata, xpred=xdata, instname1=None, instname2=None)
        K = self.compute_cov_matrix(pars, include_uncorr_error=True)
        
        # Compute version of K without intrinsic data error
        self.initialize(p0=pars, x1=xpred, xpred=xdata, instname1=instname, instname2=None)
        Ks = self.compute_cov_matrix(pars, include_uncorr_error=False)

        # Avoid overflow errors by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, linpred)
        mu = np.dot(Ks, alpha).flatten()

        return mu
    
    def compute_noise_components(self, pars, linpred, xpred=None):
        if xpred is None:
            xpred = self.data.x
        comps = {}
        for data in self.data.values():
            comps[f"GP {data.label}"] = self.compute_gp_with_error(pars=pars, linpred=linpred, instname=data.label, xpred=xpred)
        return comps
          
    def compute_corr_error(self, pars, instname, linpred, xpred=None):
        
        # Get grids
        xdata = self.data.x
        if xpred is None:
            xpred = xdata
            
        # Get K
        self.initialize(p0=pars, x1=xdata, xpred=xdata, instname1=None, instname2=None)
        K = self.compute_cov_matrix(pars, include_uncorr_error=True)

        # Avoid overflow errors by reducing the matrix.
        L = cho_factor(K)

        self.initialize(p0=pars, x1=xpred, xpred=xdata, instname1=instname, instname2=None)
        Ks = self.compute_cov_matrix(pars, include_uncorr_error=False)
            
        self.initialize(p0=pars, x1=xpred, xpred=xpred, instname1=instname, instname2=instname)
        Kss = self.compute_cov_matrix(pars, include_uncorr_error=False)
        
        B = cho_solve(L, Ks.T)
        
        gp_error = np.sqrt(np.array(np.diag(Kss - np.dot(Ks, B))).flatten())
        
        return gp_error
    
    def compute_gp_with_error(self, pars, linpred, instname, xpred=None):
        
        # Get grids
        xdata = self.data.x
        if xpred is None:
            xpred = xdata
        
        # Get K
        self.initialize(p0=pars, x1=xdata, xpred=xdata, instname1=None, instname2=None)
        K = self.compute_cov_matrix(pars, include_uncorr_error=True)
        
        # Compute version of K without intrinsic data error
        self.initialize(p0=pars, x1=xpred, xpred=xdata, instname1=instname, instname2=None)
        Ks = self.compute_cov_matrix(pars, include_uncorr_error=False)

        # Avoid overflow errors by reducing the matrix.
        L = cho_factor(K)
        alpha = cho_solve(L, linpred)
        mu = np.dot(Ks, alpha).flatten()
        
        # Kss
        self.initialize(pars, x1=xpred, xpred=xpred, instname1=instname, instname2=instname)
        Kss = self.compute_cov_matrix(pars, include_uncorr_error=False)
        
        B = cho_solve(L, Ks.T)
        
        error = np.sqrt(np.array(np.diag(Kss - np.dot(Ks, B))).flatten())
        
        return mu, error
    
    def compute_residuals(self, pars, linpred):
        residuals = np.copy(linpred)
        for data in self.data.values():
            residuals[self.data.indices[data.label]] -= self.realize(pars=pars, linpred=linpred, instname=data.label, xpred=data.t)
        return residuals
        
    
    def initialize(self, p0, x1=None, xpred=None, instname1=None, instname2=None):
        self.kernel.initialize(p0=p0, x1=x1, xpred=xpred, instname1=instname1, instname2=instname2)