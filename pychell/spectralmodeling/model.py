# Maths
import numpy as np

# pychell
import pychell.maths as pcmath

# Optimize
from optimize.parameters import BoundedParameters, BoundedParameter

##################################
#### COMPOSITE SPECTRAL MODEL ####
##################################

class SpectralForwardModel:
    """The primary container for a spectral forward model.
    """

    __slots__ = ["wls", "continuum", "lsf", "star", "tellurics", "gascell", "fringing", "oversample", "templates", "sregion"]
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################
    
    def __init__(self, wls=None, continuum=None, lsf=None,
                 star=None,
                 tellurics=None,
                 gascell=None,
                 fringing=None,
                 sregion=None,
                 oversample=8):
        """Initiate an iterative spectral forward model object.

        Args:
            wls (WavelengthSolution, optional): The wavelength solution model. Defaults to None.
            continuum (Continuum, optional): The continuum model. Defaults to None.
            lsf (LSF, optional): The LSF model. Defaults to None.
            star (AugmentedStar, optional): The Stellar model. Defaults to None.
            tellurics (Tellurics, optional): The telluric model. Defaults to None.
            gascell (GasCell, optional): The gas cell model. Defaults to None.
            fringing (FPCavityFringing, optional): The fringing model. Defaults to None.
        """

        # Oversample factor
        self.oversample = oversample
        self.sregion = sregion
        
        # Model components
        self.wls = wls
        self.continuum = continuum
        self.lsf = lsf
        self.star = star
        self.tellurics = tellurics
        self.gascell = gascell
        self.fringing = fringing
        
    def load_templates(self, data):
        if self.sregion.pixmin is None:
            dl = (self.sregion.wave_len() / len(data[0].flux)) / self.oversample
        else:
            dl = (1 / self.sregion.pix_per_wave()) / self.oversample
        wave = np.arange(self.sregion.wavemin - 1.5, self.sregion.wavemax + 1.5, dl)
        wave = wave[2:-2]
        self.templates = {"wave": wave}
        if self.star is not None:
            self.templates["star"] = self.star.load_template(self.templates["wave"])
        if self.gascell is not None:
            self.templates["gascell"] = self.gascell.load_template(self.templates["wave"])
        if self.tellurics is not None:
            #breakpoint()
            #if self.lsf is not None:
            #    wave_rel = get_wave_grid(self.sregion, dl)
            #    kernel = pcmath.gauss(wave_rel, amp, 0, self.lsf.sigma[2])
            #else:
            #    kernel = None
            self.templates["tellurics"] = self.tellurics.load_template(self.templates["wave"])
        
    def get_init_parameters(self, data):
        p0 = BoundedParameters()
        if self.wls is not None:
            p0.update(self.wls.get_init_parameters(data, self.templates, self.sregion))
        if self.lsf is not None:
            p0.update(self.lsf.get_init_parameters(data, self.templates, self.sregion))
        if self.continuum is not None:
            p0.update(self.continuum.get_init_parameters(data, self.templates, self.sregion))
        if self.star is not None:
            p0.update(self.star.get_init_parameters(data, self.templates, self.sregion))
        if self.gascell is not None:
            p0.update(self.gascell.get_init_parameters(data, self.templates, self.sregion))
        if self.tellurics is not None:
            p0.update(self.tellurics.get_init_parameters(data, self.templates, self.sregion))
        if self.fringing is not None:
            p0.update(self.fringing.get_init_parameters(data, self.templates, self.sregion))
        
        return p0
    
    ##################
    #### BUILDERS ####
    ##################
        
    def build(self, pars, data, interp=True):
            
        # Init a model
        model_flux = np.ones(len(self.templates["wave"]))

        # Star
        if self.star is not None:
            model_flux *= self.star.build(pars, self.templates)
        
        # Gas Cell
        if self.gascell is not None:
            model_flux *= self.gascell.build(pars, self.templates)
            
        # All tellurics
        if self.tellurics is not None and not self.tellurics.mask:
            model_flux *= self.tellurics.build(pars, self.templates)
        
        # Fringing from who knows what
        if self.fringing is not None:
            model_flux *= self.fringing.build(pars, self.templates)
            
        # Convolve
        if self.lsf is not None:
            model_flux = self.lsf.convolve(model_flux, pars=pars)
            
        # Continuum
        if self.continuum is not None:
            model_flux *= self.continuum.build(pars, self.templates["wave"])

        # Generate the wavelength solution of the data
        if self.wls is not None:
            data_wave = self.wls.build(pars, data)

        # Interpolate high res model onto data grid
        if interp:
            model_flux = pcmath.cspline_interp(self.templates['wave'], model_flux, data_wave)
            return data_wave, model_flux
        else:
            return templates['wave'], model_flux

    
    ###############
    #### MISC. ####
    ###############
    
    def summary(self, pars):
        s = ""
        if self.wls is not None:
            s += "--Wavelength Solution--:\n"
            for pname in self.wls.par_names:
                s += f"{pars[pname]}\n"
        if self.continuum is not None:
            s += "--Continuum--:\n"
            for pname in self.continuum.par_names:
                s += f"{pars[pname]}\n"
        if self.lsf is not None:
            s += "--LSF--:\n"
            for pname in self.lsf.par_names:
                s += f"{pars[pname]}\n"
        if self.star is not None:
            s += "--Star--:\n"
            for pname in self.star.par_names:
                s += f"{pars[pname]}\n"
        if self.tellurics is not None:
            s += "--Tellurics--:\n"
            for pname in self.tellurics.par_names:
                s += f"{pars[pname]}\n"
        if self.gascell is not None:
            s += "--Gas Cell--:\n"
            for pname in self.gascell.par_names:
                s += f"{pars[pname]}\n"
        return s

    def __repr__(self):
        s = f"Spectral Forward Model [{self.sregion}]\n"
        if self.wls is not None:
            s += f"  {self.wls}\n"
        if self.continuum is not None:
            s += f"  {self.continuum}\n"
        if self.lsf is not None:
            s += f"  {self.lsf}\n"
        if self.star is not None:
            s += f"  {self.star}\n"
        if self.tellurics is not None:
            s += f"  {self.tellurics}\n"
        if self.gascell is not None:
            s += f"  {self.gascell}\n"
        s = s[0:-1]
        return s

