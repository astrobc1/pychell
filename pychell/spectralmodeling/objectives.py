# Maths
import numpy as np

# Pychell deps
import pychell.maths as pcmath

class SpectralObjectiveFunction:
    
    def __init__(self, flag_worst=10, remove_edges=4):
        """Constructs the objective function.

        Args:
            flag_n_worst_pixels (int): Flags the worse N pixels on each attempt.
            remove_edges (int): The number of pixels on the left and right to remove.
        """
        self.flag_worst = flag_worst
        self.remove_edges = remove_edges
        
    def compute_obj(self, pars):
        raise NotImplementedError(f"Must Implement a compute_obj method for the class {self.__class__.__name__}")

    def __call__(self, *args, **kwargs):
        return self.compute_obj(*args, **kwargs)
    
    def initialize(self, spectral_model):
        self.spectral_model = spectral_model

    def mask_tellurics(self):
        self.spectral_model.mask_tellurics(self.templates['tellurics'])
        
    @property
    def p0(self):
        return self.spectral_model.p0

class RMSSpectralObjective(SpectralObjectiveFunction):
    """Objective function which returns the weighted RMS. The weights are prop. to 1 / flux_unc^2. The LSF is further enforced to be positive.
    """

    __slots__ = ["flag_worst", "remove_edges", "weight_snr"]

    def __init__(self, flag_worst=10, remove_edges=4, weight_snr=False):
        """Constructs the objective function.

        Args:
            flag_worst (int): Flags the worse N pixels on each attempt.
            remove_edges (int): The number of pixels on the left and right to ignore.
            weight_snr (bool, optional): Whether or not to include the S/N of each pixel as weights, ~ 1/snr^2.
        """
        super().__init__(flag_worst=flag_worst, remove_edges=remove_edges)
        self.weight_snr = weight_snr

    def compute_obj(self, pars, data, model):

        # Generate the forward model
        wave_model, flux_model = model.build(pars, data)

        # Weights are prop. to 1 / unc^2
        if self.weight_snr:
            weights = data.mask * data.flux / data.fluxerr**2
        else:
            weights = np.copy(data.mask)

        # Flag tellurics
        # if self.spectral_model.tellurics is not None and self.spectral_model.tellurics.mask:
        #     tell_mask = self.spectral_model.tellurics.mask_tellurics(pars, self.spectral_model.templates['tellurics'], wave_model)
        #     weights *= tell_mask

        # Compute rms ignoring bad pixels
        try:
            rms = pcmath.rmsloss(data.flux, flux_model, weights=weights, flag_worst=self.flag_worst, remove_edges=self.remove_edges)
        except:
            return 1E6
        
        # Force LSF to be positive everywhere.
        if model.lsf is not None and np.min(model.lsf.build(pars=pars)) < 0:
            rms += 1E6

        # Return final rms
        return rms
    
    def __repr__(self):
        return f"Spectral objective function with SNR weights: {self.weight_snr}"