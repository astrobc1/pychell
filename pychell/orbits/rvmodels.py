import optimize.models as optmodels
import optimize.kernels as optnoisekernels
import numpy as np
import time
import matplotlib.pyplot as plt
from numba import jit, njit, prange

class RVModel(optmodels.Model):
    """
    A Base RV Bayesian RV Model
    
    Attributes:
        planets_dict (dict): A planets dictionary containing indices (integers) as keys, and letters as values, akin to the radvel dictionary.
        data (MixedRVData): The composite RV data set.
        p0 (Parameters): The initial parameters.
        kernels (list): The list of noise kernels.
        time_base (float): The time to subtract off for the linear and quadratic gamma offsets.
    """
    
    def __init__(self, planets_dict=None, data=None, p0=None, kernel=None, time_base=None):
        """Construct an RV Model for multiple datasets.

        Args:
            planets_dict (dict): A planets dictionary containing indices (integers) as keys, and letters as values, akin to the radvel dictionary.
            data (RVData): The composite RV data set.
            p0 (Parameters): The initial parameters.
            kernel (NoiseKernel): The noise kernel.
            time_base (float): The time to subtract off for the linear and quadratic gamma offsets.
        """
        
        # Call super init
        super().__init__(data=data, p0=p0, kernel=kernel)
        
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
        vels = planet_signal(t, *planet_pars)
        
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

    def apply_offsets(self, rv_vec, pars, instname=None):
        """Apply gamma offsets (zero points only) to the data. Linear and quadratic terms are applied to the model.

        Args:
            data (MixedData): The full data object.
            rv_vec (np.ndarray): The RV data vector for all data
            pars (Parameters): The parameters.
        """
        
        # Remove the per-instrument effective zero points.
        if instname is None:
            for data in self.data.values():
                pname = "gamma_" + data.label
                rv_vec[self.data_inds[data.label]] -= pars[pname].value
        else:
            pname = "gamma_" + instname
            rv_vec -= pars[pname].value
                
        # Linear trend
        if 'gamma_dot' in pars and pars['gamma_dot'].value != 0:
            rv_vec -= pars['gamma_dot'].value * (t - self.time_base)
        
        # Quadratic trend
        if 'gamma_ddot' in pars and pars['gamma_ddot'].value != 0:
            rv_vec -= pars['gamma_ddot'].value * (t - self.time_base)**2
            
        return rv_vec

    def __repr__(self):
        return 'An RV Model'


# class RVModelGrad(RVModel, optmodels.PyMC3Model):
    
#     def __init__(self, planets_dict=None, data=None, p0=None, kernel=None, time_base=None):
#         RVModel.__init__(planets_dict=planets_dict, data=data, p0=None, kernel=None, time_base=None)
        
    

@njit
def solve_kepler(mas, ecc):
    eas = np.zeros_like(mas)
    for i in range(mas.size):
        eas[i] = _solve_kepler(mas[i], ecc)
    return eas

@njit
def _solve_kepler(ma, ecc):
    """Solve Kepler's Equation for one planet.
    Args:
        ma (float): mean anomaly
        eccarr (float): eccentricity
    Returns:
        float: The eccentric anomaly.
    """

    # Convergence criterion
    conv = 1E-12
    k = 0.85
    
    # First guess for ea
    ea = ma + np.sign(np.sin(ma)) * k * ecc
    fi = ea - ecc * np.sin(ea) - ma
    
    # Counter
    count = 0
    
    # Break when converged
    while True:
        
        # Increase counter
        count += 1
        
        # Update ea
        fip = 1 - ecc * np.cos(ea)
        fipp = ecc * np.sin(ea)
        fippp = 1 - fip
        d1 = -fi / fip
        d2 = -fi / (fip + d1 * fipp / 2.0)
        d3 = -fi / (fip + d2 * fipp / 2.0 + d2 * d2 * fippp / 6.0)
        ea_new = ea + d3
        
        # Check convergence
        fi = ea_new - ecc * np.sin(ea_new) - ma
        if fi < conv:
            break
        ea = ea_new
    
    return ea_new

@njit
def true_anomaly(t, tp, per, ecc):
    """
    Calculate the true anomaly for a given time, period, eccentricity.

    Args:
        t (np.ndarray): The times.
        tp (float): The time of periastron.
        per (float): The period of the orbit in units of t.
        ecc (float): The eccentricity of the bounded orbit.

    Returns:
        np.ndarray: true anomoly at each time
    """
    
    m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
    ea1 = solve_kepler(m, ecc)
    n1 = 1.0 + ecc
    n2 = 1.0 - ecc
    nu = 2.0 * np.arctan((n1 / n2)**0.5 * np.tan(ea1 / 2.0))

    return nu


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

    # Let a negative period be zero
    if per < 0:
        per = 1E-4
        
    # Force circular orbit if ecc is negative
    if ecc < 0:
        ecc = 0
        m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
        return k * np.cos(m + w)
    
    # Force bounded orbit if ecc > 1
    if ecc > 0.99:
        ecc = 0.99
        
    # Calculate the eccentric anomaly (ea) from the mean anomaly (ma).
    ta = true_anomaly(t, tp, per, ecc)
    rv = k * (np.cos(ta + w) + ecc * np.cos(w))

    # Return rv
    return rv

@njit
def tc_to_tp(tc, per, ecc, w):
    """
    Convert Time of Transit (time of conjunction) to Time of Periastron Passage

    Args:
        tc (float): time of transit
        per (float): period [days]
        ecc (float): eccentricity
        w (float): angle of periastron (radians)

    Returns:
        float: time of periastron passage

    """
    
    # If ecc >= 1, no tp exists
    if ecc >= 1:
        return tc

    f = np.pi / 2 - w
    ee = 2 * np.arctan(np.tan(f / 2) * np.sqrt((1 - ecc) / (1 + ecc)))  # eccentric anomaly (ee = f for ecc=0)
    tp = tc - per / (2 * np.pi) * (ee - ecc * np.sin(ee)) # time of periastron

    return tp

@njit
def tp_to_tc(tp, per, ecc, w):
    """
    Convert Time of Periastron to Time of Transit (time of conjunction).

    Args:
        tp (float): time of periastron
        per (float): period [days]
        ecc (float): eccentricity
        w (float): argument of periastron (radians)
        secondary (bool): calculate time of secondary eclipse instead

    Returns:
        float: time of inferior conjunction (time of transit if system is transiting)
    """
    
    # If ecc >= 1, no tc exists.
    if ecc >= 1:
        return tp

    f = np.pi / 2 - w                                         # true anomaly during transit
    ee = 2 * np.arctan(np.tan( f / 2) * np.sqrt((1 - ecc) / (1 + ecc)))  # eccentric anomaly

    tc = tp + per / (2 * np.pi) * (ee - ecc * np.sin(ee))         # time of conjunction

    return tc


class AbstractOrbitBasis:
    
    def __init__(self, planet_index):
        self.planet_index = planet_index
        ii = str(self.planet_index)
        self.pnames = [name + ii for name in self.names]
        
    def to_standard(self, pars):
        pass
    
    @classmethod
    def from_standard(cls, pars):
        pass
        
        
class StandardOrbitBasis(AbstractOrbitBasis):
    
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
    
    names = ["per", "tc", "ecc", "w", "k"]
    
    def to_standard(self, pars):
        ii = str(self.planet_index)
        per = pars["per" + ii].value
        tc = pars["tc" + ii].value
        ecc = pars["ecc" + ii].value
        w = pars["w" + ii].value
        k = pars["k" + ii].value
        tp = tc_to_tp(tc, per, ecc, w)
        return (per, tp, ecc, w, k)
    
    def from_standard(self, pars):
        ii = str(self.planet_index)
        per = pars["per" + ii].value
        tp = pars["tp" + ii].value
        ecc = pars["ecc" + ii].value
        w = pars["w" + ii].value
        k = pars["k" + ii].value
        tc = tp_to_tc(tp, per, ecc, w)
        return (per, tc, ecc, w, k)
    

class TCSQEOrbitBasis(AbstractOrbitBasis):
    
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
        tp = tc_to_tp(tc, per, ecc, w)
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
        tc = tp_to_tc(tp, per, ecc, w)
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
        tp = tc_to_tp(tc, per, ecc, w)
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
        tc = tp_to_tc(tp, per, ecc, w)
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
        

        