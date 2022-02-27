name = 'spectralmodeling'

# Spectral region for 1d forward modeling
from .sregion import SpecRegion1d

# Main Spectral model object for 1d forward modeling
from .model import SpectralForwardModel

# Model components for 1d forward modeling
from .component import SpectralModelComponent1d
from .star import *
from .lsf import *
from .gascell import *
from .tellurics import *
from .continuum import *
from .wavelength import *

# Objectives for 1d forward modeling
from .objectives import SpectralObjectiveFunction, RMSSpectralObjective

# Template Augmenting (lazy)
from .augmenters import TemplateAugmenter, WeightedMeanAugmenter

# Main problem object for 1d forward modeling
from .problem import SpectralRVProblem

# Barycenter corrections
from .barycenter import compute_barycenter_corrections #, compute_barycenter_corrections_weighted

# Lazy imports for other method based files
from .fitting import *
from .plotting import *
from .rvcalc import *
from .postplayground import *