# Maths
import numpy as np

# optimize deps
from optimize.kernels import CorrelatedNoiseKernel

# pychell
import pychell.maths as pcmath

class ChromaticKernelJ1(CorrelatedNoiseKernel):
    
    def __init__(self, data, par_names):
        super().__init__(data=data, par_names=par_names)
        self.instname_vec = self.data.gen_instname_vec()
    
    def compute_cov_matrix(self, pars):
        
        # Number of instruments
        n_instruments = len(self.data)
        
        # Alias params
        amp_matrix = self.gen_amp_matrix(pars)
        exp_length = pars[self.par_names[n_instruments]].value
        per_length = pars[self.par_names[n_instruments + 1]].value
        per = pars[self.par_names[n_instruments + 2]].value

        # Construct QP terms
        decay_term = -0.5 * (self.dist_matrix / exp_length)**2
        periodic_term = -0.5 * (1 / per_length)**2 * np.sin(np.pi * self.dist_matrix / per)**2
        
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
    
    def initialize(self, p0, x1=None, xpred=None, instname1=None, instname2=None):
        if x1 is None:
            x1 = self.data.x
        if xpred is None:
            xpred = x1
        self.dist_matrix = optmath.compute_stationary_dist_matrix(x1, xpred)
        self.instname1 = instname1
        self.instname2 = instname2
        
        

        
class ChromaticKernelJ2(CorrelatedNoiseKernel):
    
    def __init__(self, data, par_names, wavelength0=550):
        super().__init__(data=data, par_names=par_names)
        self.wave_vec = self.data.gen_wave_vec()
        self.n_wavelengths = len(np.unique(self.wave_vec))
        self.wavelength0 = wavelength0
    
    def compute_cov_matrix(self, pars):
        
        # Alias params
        gp_amp_0 = pars[self.par_names[0]].value
        gp_amp_scale = pars[self.par_names[1]].value
        exp_length = pars[self.par_names[2]].value
        per_length = pars[self.par_names[3]].value
        per = pars[self.par_names[4]].value

        # Construct QP terms
        decay_term = -0.5 * (self.dist_matrix / exp_length)**2
        periodic_term = -0.5 * (1 / per_length)**2 * np.sin(np.pi * self.dist_matrix / per)**2
        
        # Construct full cov matrix
        K = gp_amp_0**2 * (self.wavelength0 / np.sqrt(self.wave_matrix))**(2 * gp_amp_scale) * np.exp(decay_term + periodic_term)
        
        return K
    
    def gen_wave_matrix(self):
        
        # The current shape of the covariance matrix
        n1, n2 = self.dist_matrix.shape
        
        # Fill each
        if self.wave1 is not None:
            wave_vec1 = np.full(n1, self.wave1)
        else:
            wave_vec1 = self.wave_vec
        if self.wave2 is not None:
            wave_vec2 = np.full(n2, self.wave2)
        else:
            wave_vec2 = self.wave_vec
                
        # Outer product: A_ij = a1 * a2 and A_ij = A_ji
        W = np.outer(wave_vec1, wave_vec2)
        
        return W
    
    def initialize(self, p0, x1=None, xpred=None, wave1=None, wave2=None):
        if x1 is None:
            x1 = self.data.x
        if xpred is None:
            xpred = x1
        self.dist_matrix = optmath.compute_stationary_dist_matrix(x1, xpred)
        self.wave1 = wave1
        self.wave2 = wave2
        self.wave_matrix = self.gen_wave_matrix()



class ChromaticKernelJ3(CorrelatedNoiseKernel):
    
    def __init__(self, data, par_names):
        super().__init__(data=data, par_names=par_names)
        self.instname_vec = self.data.gen_instname_vec()
        self.wave_vec = self.data.gen_wave_vec()
    
    def compute_cov_matrix(self, pars):
        
        # Number of instruments
        n_instruments = len(self.data)
        
        # Alias params
        amp_matrix = self.gen_amp_matrix(pars)
        exp_length = pars[self.par_names[n_instruments]].value
        per_length = pars[self.par_names[n_instruments + 1]].value
        per = pars[self.par_names[n_instruments + 2]].value
        wave_corr = pars[self.par_names[n_instruments + 3]].value

        # Construct QP terms
        decay_term = -0.5 * (self.dist_matrix / exp_length)**2
        periodic_term = -0.5 * (1 / per_length)**2 * np.sin(np.pi * self.dist_matrix / per)**2

        # Wave corr
        wave_term = -0.5 * (self.wave_diffs / wave_corr)**2
        
        # Construct full cov matrix
        K = amp_matrix * np.exp(decay_term + periodic_term + wave_term)

        
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

    def gen_wave_matrix(self):
        
        # The current shape of the covariance matrix
        n1, n2 = self.dist_matrix.shape
        
        # Fill each
        if self.wave1 is not None:
            wave_vec1 = np.full(n1, self.wave1)
        else:
            wave_vec1 = self.wave_vec
        if self.wave2 is not None:
            wave_vec2 = np.full(n2, self.wave2)
        else:
            wave_vec2 = self.wave_vec
                
        W = pcmath.outer_diff(wave_vec1, wave_vec2)
        
        return W
    
    def initialize(self, p0, x1=None, xpred=None, instname1=None, instname2=None):
        if x1 is None:
            x1 = self.data.x
        if xpred is None:
            xpred = x1
        self.dist_matrix = optmath.compute_stationary_dist_matrix(x1, xpred)
        self.instname1 = instname1
        self.instname2 = instname2
        self.wave1 = None if instname1 is None else self.data[instname1].wavelength
        self.wave2 = None if instname2 is None else self.data[instname2].wavelength
        self.wave_diffs = self.gen_wave_matrix()