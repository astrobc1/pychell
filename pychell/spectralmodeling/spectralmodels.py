# Maths
import numpy as np
from scipy.special import eval_legendre

# pychell
import pychell.maths as pcmath
from pychell.maths import cspline_interp

# Optimize
from optimize.models import Model
from optimize.knowledge import BoundedParameters, BoundedParameter

#########################
#### SPECTRAL REGION ####
#########################

class SpectralRegion:
    
    __slots__ = ['pixmin', 'pixmax', 'wavemin', 'wavemax', 'label', 'data_inds']
    
    def __init__(self, pixmin, pixmax, wavemin, wavemax, label=None):
        self.pixmin = pixmin
        self.pixmax = pixmax
        self.wavemin = wavemin
        self.wavemax = wavemax
        self.label = label
        self.data_inds = np.arange(self.pixmin, self.pixmax + 1).astype(int)
        
    def __len__(self):
        return self.pix_len()
    
    def wave_len(self):
        return self.wavemax - self.wavemin
    
    def pix_len(self):
        return self.pixmax - self.pixmin + 1
        
    def pix_within(self, pixels, pad=0):
        good = np.where((pixels >= self.pixmin - pad) & (pixels <= self.pixmax + pad))[0]
        return good
        
    def wave_within(self, waves, pad=0):
        good = np.where((waves >= self.wavemin - pad) & (waves <= self.wavemax + pad))[0]
        return good
    
    def midwave(self):
        return self.wavemin + self.wave_len() / 2
    
    def midpix(self):
        return self.pixmin + self.pix_len() / 2
        
    def pix_per_wave(self):
        return (self.pixmax - self.pixmin) / (self.wavemax - self.wavemin)
    
    def __repr__(self):
        return f"Spectral Region: Pix: ({self.pixmin}, {self.pixmax}) Wave: ({self.wavemin}, {self.wavemax})"


####################
#### COMPONENTS ####
####################

from .spectral_components import *

#########################
#### COMPOSITE MODEL ####
#########################

from .composite_spectralmodels import *