# Maths
import numpy as np

# optimize deps
from optimize.kernels import StationaryNoiseKernel

# pychell
import pychell.maths as pcmath

class ChromaticKernelJ1(StationaryNoiseKernel):
    
    def compute_cov_matrix(self, pars, x1, x2, amp_vec1, amp_vec2):

        # The number of instruments
        n_instruments = len([par for par in pars if par.startswith("gp_amp")])

        # Distance matrix
        dist_matrix = self.compute_dist_matrix(x1, x2)

        # Wave matrix
        amp_matrix = self.gen_amp_matrix(amp_vec1, amp_vec2)
        
        # Alias params
        exp_length = pars[self.par_names[n_instruments]].value
        per_length = pars[self.par_names[n_instruments+1]].value
        per = pars[self.par_names[n_instruments+2]].value

        # Construct QP terms
        decay_term = -0.5 * (dist_matrix / exp_length)**2
        periodic_term = -0.5 * (1 / per_length)**2 * np.sin(np.pi * dist_matrix / per)**2
        
        # Construct full cov matrix
        K = amp_matrix**2 * np.exp(decay_term + periodic_term)
        
        return K

    def gen_amp_matrix(self, amp_vec1, amp_vec2):
        return np.sqrt(np.outer(amp_vec1, amp_vec2))

class ChromaticKernelJ2(StationaryNoiseKernel):

    par_names = None

    def __init__(self, par_names, wavelength0):
        super().__init__(par_names)
        self.wavelength0 = wavelength0
    
    def compute_cov_matrix(self, pars, x1, x2, wave_vec1, wave_vec2):

        # Distance matrix
        dist_matrix = self.compute_dist_matrix(x1, x2)

        # Wave matrix
        wave_matrix = self.gen_wave_matrix(wave_vec1, wave_vec2)
        
        # Alias params
        gp_amp_0 = pars[self.par_names[0]].value
        gp_amp_scale = pars[self.par_names[1]].value
        exp_length = pars[self.par_names[2]].value
        per_length = pars[self.par_names[3]].value
        per = pars[self.par_names[4]].value

        # Construct QP terms
        decay_term = -0.5 * (dist_matrix / exp_length)**2
        periodic_term = -0.5 * (1 / per_length)**2 * np.sin(np.pi * dist_matrix / per)**2
        
        # Construct full cov matrix
        K = gp_amp_0**2 * (self.wavelength0 / np.sqrt(wave_matrix))**(2 * gp_amp_scale) * np.exp(decay_term + periodic_term)
        
        return K

    def gen_wave_matrix(self, wave_vec1, wave_vec2):
        return np.outer(wave_vec1, wave_vec2)