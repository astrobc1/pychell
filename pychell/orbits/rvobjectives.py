from scipy.linalg import cho_factor, cho_solve
import numpy as np
from optimize.objectives import GaussianLikelihood, Posterior

#######################
#### RV LIKELIHOOD ####
#######################

class RVLikelihood(GaussianLikelihood):
    pass


######################
#### RV POSTERIOR ####
######################

class RVPosterior(Posterior):
    """A class for RV Posteriors.
    """
    pass