import optimize.models as optmodels
import optimize.kernels as optnoisekernels
import numpy as np
import time
import matplotlib.pyplot as plt
from numba import jit, njit, prange
import pychell.orbits.planetmath as planetmath

class RVModel(optmodels.Model):
    """
    A Base RV Model intended for Bayesian inference.
    
    Attributes:
        planets_dict (dict): A planets dictionary containing indices (integers) as keys, and sub dictionaries as values. Each sub dict is composed of a label key with a character value (i.e., "label": "b" for the first planet) as well as a basis key with a valid orbit basis value. (i.e., "basis": <TCOrbitBasis instance>).
        data (CompositeRVData): The composite RV data set.
        p0 (Parameters): The initial parameters.
        time_base (float): The time to subtract off for the linear and quadratic gamma offsets.
    """
    
    def __init__(self, planets_dict=None, data=None, p0=None, time_base=None):
        """Construct an RV Model for multiple datasets.

        Args:
            planets_dict (dict): A planets dictionary containing indices (integers) as keys, and sub dictionaries as values. Each sub dict is composed of a label key with a character value (i.e., "label": "b" for the first planet) as well as a basis key with a valid orbit basis value. (i.e., "basis": <TCOrbitBasis instance>).
            data (CompositeRVData): The composite RV data set.
            p0 (Parameters): The initial parameters.
            time_base (float): The time to subtract off for the linear and quadratic gamma offsets.
        """
        
        # Call super init
        super().__init__(data=data, p0=p0)
        
        # Store extra attributes
        self.planets_dict = planets_dict
        if time_base is None:
            self.time_base = np.nanmean(self.data.get_vec(key='x'))
            
        self.data_t = self.data.get_vec('t')
        self.data_rv = self.data.get_vec('rv')
        self.data_rverr = self.data.get_vec('rverr')
        self.data_inds = {}
        for data in self.data.values():
            self.data_inds[data.label] = self.data.get_inds(data.label)
        
    @property
    def n_planets(self):
        return len(self.planets_dict)
    
    def build_planet(self, pars, t, planet_index):
        """Builds a model for a single planet.

        Args:
            pars (Parameters): The parameters.
            t (np.ndarray): The times.
            planet_dict (dict): The planet dictionary.

        Returns:
            np.ndarray: The RV model for this planet
        """
        
        # Get the planet parameters on the standard basis
        planet_pars = self.planets_dict[planet_index]["basis"].to_standard(pars)
        
        # Build and return planet signal
        vels = self.planet_signal(t, *planet_pars)
        
        # Return vels
        return vels
    
    def build_planets(self, pars, t):
        """Builds a model including all planets.

        Args:
            t (np.ndarray): The times.
            pars (Parameters): The parameters.

        Returns:
            np.ndarray: The RV model for this planet
        """
        _model = np.zeros_like(t)
        for planet_index in self.planets_dict:
            _model += self.build_planet(pars, t, planet_index)
        return _model
    
    def build_without_planet(self, pars, t, planet_index):
        """Builds the model without a planet.

        Args:
            pars (Parameters): The parameters.
            t (np.ndarray): The time vector.
            planet_index (int): The index of the planet to not consider.

        Returns:
            np.ndarray: The model without the designated planet.
        """
        model_arr_full = self._builder(pars, t)
        planet_signal = self.build_planet(pars, t, planet_index)
        return model_arr_full - planet_signal
    
    def build_trend_zero(self, pars, t, instname=None):
        
        # Zeros
        trend_zero = np.zeros(t.size)
        
        # Per-instrument zero points
        if instname is None:
            for data in self.data.values():
                pname = "gamma_" + data.label
                inds = self.data_inds[data.label]
                trend_zero[inds] = pars[pname].value
        else:
            pname = "gamma_" + instname
            trend_zero += pars[pname].value
        
        return trend_zero

    def build_trend_global(self, pars, t):
        
        # Zeros
        trend_global = np.zeros(t.size)
                
        # Linear trend
        if 'gamma_dot' in pars and pars['gamma_dot'].value != 0:
            trend_global += pars['gamma_dot'].value * (t - self.time_base)
        
        # Quadratic trend
        if 'gamma_ddot' in pars and pars['gamma_ddot'].value != 0:
            trend_global += pars['gamma_ddot'].value * (t - self.time_base)**2
            
        return trend_global
        
    def _builder(self, pars, t):
        
        # All planets
        _model = self.build_planets(pars, t)
        
        # Return model
        return _model
        
    def build(self, pars):
        """Builds the model using the specified parameters on the data grid.

        Args:
            pars (Parameters): The parameters to use

        Returns:
            np.ndarray: The full model.
        """
        _model = self._builder(pars, self.data_t)
        return _model

    def apply_offsets(self, rv_vec, pars, t=None, instname=None):
        """Apply gamma offsets (zero points only) to the data. Linear and quadratic terms are applied to the model.

        Args:
            data (MixedData): The full data object.
            rv_vec (np.ndarray): The RV data vector for all data
            pars (Parameters): The parameters.
        """
        if t is None and instname is None:
            t = self.data_t
        if t is None and instname is not None:
            t = self.data[instname].t
        trend_zero = self.build_trend_zero(pars, t=t, instname=instname)
        trend_global = self.build_trend_global(pars, t=t)
        rv_vec -= (trend_zero + trend_global)
        return rv_vec

    def __repr__(self):
        return 'An RV Model'

    @staticmethod
    @njit
    def planet_signal(t, per, tp, ecc, w, k):
        """Computes the RV signal of one planet for a given time vector.

        Args:
            t (np.ndarray): The times in units of per.
            k (float): The RV semi-amplitude.
            per (float): The period of the orbit in units of t.
            tc (float): The time of conjunction.
            ecc (float): The eccentricity of the bounded orbit.
            w (float): The angle of periastron
            tp (float): The time of perisatron

        Returns:
            np.ndarray: The rv signal for this planet.
        """

        # Circular orbit
        if ecc == 0.0:
            m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
            return k * np.cos(m + w)

        # Period must be positive
        if per <= 0:
            per = 1E-6
            
        # Force circular orbit if ecc is negative
        if ecc < 0:
            ecc = 0
            m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
            return k * np.cos(m + w)
        
        # Force bounded orbit if ecc > 1
        if ecc > 0.99:
            ecc = 0.99
            
        # Calculate the eccentric anomaly (ea) from the mean anomaly (ma). Requires solving kepler's eq. if ecc>0.
        ta = planetmath.true_anomaly(t, tp, per, ecc)
        rv = k * (np.cos(ta + w) + ecc * np.cos(w))

        # Return rv
        return rv

    @staticmethod
    def disable_planet_pars(pars, planets_dict, planet_index):
        """Disables (sets vary=False) in-place for the planet parameters corresponding to planet_index.

        Args:
            pars (Parameters): The parameters.
            planets_dict (dict): The planets dict.
            planet_index (int): The index to disable.
        """
        for par in pars.values():
            for planet_par_name in planets_dict[planet_index]["basis"].names:
                if par.name == planet_par_name + str(planet_index):
                    pars[par.name].vary = False

class AbstractOrbitBasis:
    """An abstract orbit basis class, not useful on its own. Each method must define to_standard and from_standard below.
    
    Attributes:
        planet_index (int): The index of this planet in the planets dictionary.
        pnames (list[str]): A list of the parameter names for this planet and basis combination.
    """
    
    def __init__(self, planet_index):
        """Constructor for most bases.

        Args:
            planet_index (int): The index of this planet in the planets dictionary.
        """
        self.planet_index = planet_index
        ii = str(self.planet_index)
        self.pnames = [name + ii for name in self.names]
        
    def to_standard(self, pars):
        """Converts the parameters to the standard basis: per, tp, ecc, w, k.

        Args:
            pars (Parameters): The input parameters.
            
        Returns:
        (tuple): tuple containing:
            float: Period.
            float: Time of periastron.
            float: Eccentricity.
            float: Angle of periastron.
            float: Semi-amplitude.
        """
        raise NotImplementedError(f"Must implement a to_standard method for basis class {self.__class__}")
    
    @classmethod
    def from_standard(cls, pars):
        """Converts the parameters to this basis from the standard basis: per, tp, ecc, w, k.

        Args:
            pars (Parameters): The input parameters.
            
        Returns:
            tuple: The basis parameters. See the class attribute names for each.
        """
        raise NotImplementedError(f"Must implement a from_standard method for class {self.__class__}")

class StandardOrbitBasis(AbstractOrbitBasis):
    """The standard orbit basis: per, tp, ecc, w, k.
    """
    
    names = ["per", "tp", "ecc", "w", "k"]
    
    def to_standard(self, pars):
        ii = str(self.planet_index)
        per = pars["per" + ii].value
        tp = pars["tp" + ii].value
        ecc = pars["ecc" + ii].value
        w = pars["w" + ii].value
        k = pars["k" + ii].value
        return (per, tp, ecc, w, k)
    
    def from_standard(self, pars):
        return self.to_standard(pars)

class TCOrbitBasis(AbstractOrbitBasis):
    """A basis utilizing tc over tp: per, tc, ecc, w, k.
    """
    
    names = ["per", "tc", "ecc", "w", "k"]
    
    def to_standard(self, pars):
        ii = str(self.planet_index)
        per = pars["per" + ii].value
        tc = pars["tc" + ii].value
        ecc = pars["ecc" + ii].value
        w = pars["w" + ii].value
        k = pars["k" + ii].value
        tp = planetmath.tc_to_tp(tc, per, ecc, w)
        return (per, tp, ecc, w, k)
    
    def from_standard(self, pars):
        ii = str(self.planet_index)
        per = pars["per" + ii].value
        tp = pars["tp" + ii].value
        ecc = pars["ecc" + ii].value
        w = pars["w" + ii].value
        k = pars["k" + ii].value
        tc = planetmath.tp_to_tc(tp, per, ecc, w)
        return (per, tc, ecc, w, k)

class TCSQEOrbitBasis(AbstractOrbitBasis):
    """The preferred basis when the angle of periastron is unknown: per, tc, sqrt(ecc)*cos(w), sqrt(ecc)*sin(w), k.
    """
    
    names = ["per", "tc", "sqecosw", "sqesinw", "k"]
    
    def to_standard(self, pars):
        ii = str(self.planet_index)
        per = pars["per" + ii].value
        k = pars["k" + ii].value
        tc = pars["tc" + ii].value
        sqecosw = pars["sqecosw" + ii].value
        sqesinw = pars["sqesinw" + ii].value
        w = np.arctan2(sqesinw, sqecosw)
        ecc = sqecosw**2 + sqesinw**2
        tp = planetmath.tc_to_tp(tc, per, ecc, w)
        return (per, tp, ecc, w, k)
        
    def from_standard(self, pars):
        ii = str(self.planet_index)
        per = pars["per" + ii].value
        k = pars["k" + ii].value
        tp = pars["tp" + ii].value
        ecc = pars["ecc" + ii].value
        w = pars["w" + ii].value
        eccsq = np.sqrt(ecc)
        sqecosw = eccsq * np.cos(w)
        sqesinw = eccsq * np.sin(w)
        tc = planetmath.tp_to_tc(tp, per, ecc, w)
        return (per, tc, sqecosw, sqesinw, k)
    
    def convert_unc_to_standard(self, unc_dict):
        ii = str(self.planet_index)
        per_unc = unc_dict["per" + ii]
        k_unc = unc_dict["k" + ii]
        tp_unc = unc_dict["tp" + ii]
        sqecosw_unc = unc_dict["sqecosw" + ii]
        sqesinw_unc = unc_dict["sqesinw" + ii]
        
        ecc_unc = np.sqrt((2 * sqecosw_unc + sqesinw_unc**2)**2 * sqecosw_unc**2 + \
                          (sqecosw_unc**2 + 2 * sqesinw_unc)**2 * sqesinw_unc**2)
        
        w_unc = np.sqrt((sqesinw_unc / (sqecosw_unc**2 + sqesinw_unc**2))**2 * sqecosw_unc**2 + \
                          (sqew_unc / (sqecosw_unc**2 + sqesinw_unc**2))**2 * sqesinw_unc**2)
        
        return (per_unc, tp_unc, ecc_unc, w_unc, k_unc)

class TCEOrbitBasis(AbstractOrbitBasis):
    
    names = ["per", "tc", "cosw", "sinw", "k"]
    
    def to_standard(self, pars):
        ii = str(self.planet_index)
        per = pars["per" + ii].value
        k = pars["k" + ii].value
        ecc = pars["ecc" + ii].value
        tc = pars["tc" + ii].value
        cosw = pars["cosw" + ii].value
        sinw = pars["sinw" + ii].value
        w = np.arctan2(sinw, cosw)
        tp = planetmath.tc_to_tp(tc, per, ecc, w)
        return (per, tp, ecc, w, k)
        
    def from_standard(self, pars):
        ii = str(self.planet_index)
        per = pars["per" + ii].value
        k = pars["k" + ii].value
        tp = pars["tp" + ii].value
        ecc = pars["ecc" + ii].value
        w = pars["w" + ii].value
        cosw = np.cos(w)
        sinw = np.sin(w)
        tc = planetmath.tp_to_tc(tp, per, ecc, w)
        return (per, tc, ecc, cosw, sinw, k)
    
    def convert_unc_to_standard(self, unc_dict):
        ii = str(self.planet_index)
        per_unc = unc_dict["per" + ii]
        k_unc = unc_dict["k" + ii]
        tp_unc = unc_dict["tp" + ii]
        sqecosw_unc = unc_dict["sqecosw" + ii]
        sqesinw_unc = unc_dict["sqesinw" + ii]
        
        ecc_unc = np.sqrt((2 * sqecosw_unc + sqesinw_unc**2)**2 * sqecosw_unc**2 + \
                          (sqecosw_unc**2 + 2 * sqesinw_unc)**2 * sqesinw_unc**2)
        
        w_unc = np.sqrt((sqesinw_unc / (sqecosw_unc**2 + sqesinw_unc**2))**2 * sqecosw_unc**2 + \
                          (sqew_unc / (sqecosw_unc**2 + sqesinw_unc**2))**2 * sqesinw_unc**2)
        
        return (per_unc, tp_unc, ecc_unc, w_unc, k_unc)
