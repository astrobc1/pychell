
######################
#### MISC. MODELS ####
######################

#### Fringing ####
class FPCavityFringing(SpectralComponent):
    """A basic Fabry-Perot cavity model for fringing in spectrographs like iSHELL and NIRSPEC.
    """
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self):

        # Super
        super().__init__()

        self.par_names = ['fringing_logd', 'fringing_fin']

    def init_parameters(self, data):
        
        pars = BoundedParameters()
        
        pars[self.par_names[0]] = BoundedParameter(value=self.blueprint['logd'][1],
                                                       vary=True,
                                                       lower_bound=self.blueprint['logd'][0],
                                                       upper_bound=self.blueprint['logd'][2])
        pars[self.par_names[1]] = BoundedParameter(value=self.blueprint['fin'][1],
                                                       vary=True,
                                                       lower_bound=self.blueprint['fin'][0],
                                                       upper_bound=self.blueprint['fin'][2])
        
        return pars

    ##################
    #### BUILDERS ####
    ##################

    def build(self, pars, wave_final):
        d = np.exp(pars[self.par_names[0]].value)
        fin = pars[self.par_names[1]].value
        theta = (2 * np.pi / wave_final) * d
        fringing = 1 / (1 + fin * np.sin(theta / 2)**2)
        return fringing
