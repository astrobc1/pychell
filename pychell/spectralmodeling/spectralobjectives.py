# Maths
import numpy as np

# Pychell deps
import pychell.maths as pcmath

class SpectralObjectiveFunction:
    
    def __init__(self, flag_n_worst_pixels=10, remove_edges=4):
        """Constructs the objective function.

        Args:
            flag_n_worst_pixels (int): Flags the worse N pixels on each attempt.
            remove_edges (int): The number of pixels on the left and right to remove.
        """
        self.flag_n_worst_pixels = flag_n_worst_pixels
        self.remove_edges = remove_edges
        
    def compute_obj(self, pars):
        raise NotImplementedError(f"Must Implement a compute_obj method for the class {self.__class__.__name__}")
    
    def initialize(self, spectral_model):
        self.spectral_model = spectral_model
        
    @property
    def p0(self):
        return self.spectral_model.p0

class WeightedSpectralUncRMS(SpectralObjectiveFunction):
    """Objective function which returns the weighted RMS. The weights are prop. to 1 / flux_unc^2. The LSF is further enforced to be positive.
    """

    def __init__(self, flag_n_worst_pixels=10, remove_edges=4, use_flux_unc=True):
        """Constructs the objective function.

        Args:
            flag_n_worst_pixels (int): Flags the worse N pixels on each attempt.
            remove_edges (int): The number of pixels on the left and right to remove.
            use_flux_unc (bool, optional): Whether or not to include the flux uncertainty as weights, ~ 1/unc^2.
        """
        super().__init__(flag_n_worst_pixels=flag_n_worst_pixels, remove_edges=remove_edges)
        self.use_flux_unc = use_flux_unc

    def compute_obj(self, pars):
        
        # Alias the data
        data_flux = np.copy(self.spectral_model.data.flux)

        # Generate the forward model
        wave_model, flux_model = self.spectral_model.build(pars)

        # Weights are prop. to 1 / unc^2
        if self.use_flux_unc:
            weights = self.spectral_model.data.mask / self.spectral_model.data.flux_unc**2
        else:
            weights = np.copy(self.spectral_model.data.mask)

        # Compute rms ignoring bad pixels
        rms = pcmath.rmsloss(data_flux, flux_model, weights=weights, flag_worst=self.flag_n_worst_pixels, remove_edges=self.remove_edges)
        
        # Force LSF to be positive everywhere.
        if np.min(self.spectral_model.lsf.build(pars)) < 0:
            rms += 1E6

        # Return final rms
        return rms
    
    def __repr__(self):
        return "Spectral objective function: RMS weighted by 1 / flux_unc^2"