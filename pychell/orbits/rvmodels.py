import optimize.models as optmodels
import optimize.kernels as optnoisekernels
import numpy as np
import time
import matplotlib.pyplot as plt
from numba import jit, njit

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
    
    planet_par_base_names = ['per', 'tc', 'ecc', 'w', 'k']
    
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
        # Alias pars
        ii = str(planet_index)
        k = pars["k" + ii].value
        per = pars["per" + ii].value
        tc = pars["tc" + ii].value
        ecc = pars["ecc" + ii].value
        omega = pars["w" + ii].value
        
        # Convert to tp
        tp = tc_to_tp(tc, per, ecc, omega)
        
        # Build and return planet signal
        vels = planet_signal(t, k, per, ecc, omega, tp)
        
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
        
        # Planets
        _model = self.build_planets(pars, t)
        
        # Linear trend
        if pars['gamma_dot'].vary:
            _model += pars['gamma_dot'].value * (t - self.time_base)
        
        # Quadratic trend
        if pars['gamma_ddot'].vary:
            _model += pars['gamma_ddot'].value * (t - self.time_base)**2
        
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
    
    def apply_offsets(self, rv_vec, pars):
        """Apply gamma offsets (zero points only) to the data. Linear and quadratic terms are applied to the model.

        Args:
            data (MixedData): The full data object.
            rv_vec (np.ndarray): The RV data vector for all data
            pars (Parameters): The parameters.
        """
        for instname in self.data:
            rv_vec[self.data_inds[instname]] -= pars["gamma_" + instname].value
        return rv_vec
    
    def data_only_planet(self, pars, planet_index):
        """Removes the full model from the data except for one planet.

        Args:
            pars (Parameters): The parameters.
            planet_index (int): The planet index to keep in the data.

        Returns:
            dict: The modified data as a dictionary, where keys are the labels, and values are numpy arrays.
        """
        mod_data = {}
        model_array_without_planet = self.build_without_planet(pars, self.data_t, planet_index=planet_index)
        
        if self.has_gp:
            residuals = self.residuals_before_kernel(pars)
            errors = self.kernel.compute_data_errors(pars)
            gpmu = self.kernel.realize(pars, residuals=residuals, return_unc=False)
        else:
            gpmu = np.zeros_like(self.data_t)
        for data in self.data.values():
            # For each data set, subtract off the model without the above planet, the offset, and the gp.
            _mod_data = data.rv - model_array_without_planet[self.data_inds[data.label]] - pars["gamma_" + data.label].value - gpmu[self.data_inds[data.label]]
            mod_data[data.label] = _mod_data
        return mod_data
    
    def residuals_before_kernel(self, pars):
        """Computes the residuals without subtracting off the best fit noise kernel.

        Args:
            pars (Parameters): The parameters to use.

        Returns:
            np.ndarray: The residuals.
        """
        model_arr = self.build(pars)
        data_arr = np.copy(self.data_rv)
        data_arr = self.apply_offsets(data_arr, pars)
        residuals = data_arr - model_arr
        return residuals

    def __repr__(self):
        return 'An RV Model'



def solve_kepler(Marr, eccarr):
    """Solve Kepler's Equation. THIS CODE IS FROM RADVEL.
    Args:
        Marr (np.ndarray): input Mean anomaly
        eccarr (np.ndarray): eccentricity
    Returns:
        np.ndarray: The eccentric anomalies.
    """

    conv = 1.0E-12  # convergence criterion
    k = 0.85
    Earr = Marr + np.sign(np.sin(Marr)) * k * eccarr  # first guess at E
    # fiarr should go to zero when converges
    fiarr = ( Earr - eccarr * np.sin(Earr) - Marr)
    convd = np.where(np.abs(fiarr) > conv)[0]  # which indices have not converged
    nd = len(convd)  # number of unconverged elements
    count = 0
    while nd > 0:  # while unconverged elements exist
        
        count += 1

        M = Marr[convd]  # just the unconverged elements ...
        ecc = eccarr[convd]
        E = Earr[convd]

        fi = fiarr[convd]  # fi = E - e*np.sin(E)-M    ; should go to 0
        fip = 1 - ecc * np.cos(E)  # d/dE(fi) ;i.e.,  fi^(prime)
        fipp = ecc * np.sin(E)  # d/dE(d/dE(fi)) ;i.e.,  fi^(\prime\prime)
        fippp = 1 - fip  # d/dE(d/dE(d/dE(fi))) ;i.e.,  fi^(\prime\prime\prime)

        # first, second, and third order corrections to E
        d1 = -fi / fip
        d2 = -fi / (fip + d1 * fipp / 2.0)
        d3 = -fi / (fip + d2 * fipp / 2.0 + d2 * d2 * fippp / 6.0)
        E = E + d3
        Earr[convd] = E
        fiarr = ( Earr - eccarr * np.sin( Earr ) - Marr) # how well did we do?
        convd = np.abs(fiarr) > conv  # test for convergence
        nd = np.where(convd)[0].size
        
    if Earr.size > 1:
        return Earr
    else:
        return Earr[0]


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

    # f in Murray and Dermott p. 27
    m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
    eccarr = np.zeros(t.size) + ecc
    e1 = solve_kepler(m, eccarr)
    n1 = 1.0 + ecc
    n2 = 1.0 - ecc
    nu = 2.0 * np.arctan((n1 / n2)**0.5 * np.tan(e1 / 2.0))

    return nu


def planet_signal(t, k, per, ecc, omega, tp):
    """Computes the RV signal of one planet for a given time vector.

    Args:
        t (np.ndarray): The times in units of per.
        k (float): The RV semi-amplitude.
        per (float): The period of the orbit in units of t.
        tc (float): The time of conjunction.
        ecc (float): The eccentricity of the bounded orbit.
        omega (float): The angle of periastron
        tp (float): The time of perisatron

    Returns:
        np.ndarray: The rv signal for this planet.
    """

    # Circular orbits are easy
    if ecc == 0.0:
        m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
        return k * np.cos(m + omega)

    if per < 0:
        per = 1E-4
    if ecc < 0:
        ecc = 0
    if ecc > 0.99:
        ecc = 0.99
        
    # Calculate the approximate eccentric anomaly, E1, via the mean anomaly  M.
    nu = true_anomaly(t, tp, per, ecc)
    rv = k * (np.cos(nu + omega) + ecc * np.cos(omega))

    # Return rv
    return rv

@jit
def tc_to_tp(tc, per, ecc, omega):
    """
    Convert Time of Transit (time of conjunction) to Time of Periastron Passage

    Args:
        tc (float): time of transit
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): longitude of periastron (radians)

    Returns:
        float: time of periastron passage

    """
    
    # If ecc >= 1, no tp exists
    if ecc >= 1:
        return tc

    f = np.pi / 2 - omega
    ee = 2 * np.arctan(np.tan(f / 2) * np.sqrt((1 - ecc)/(1 + ecc)))  # eccentric anomaly
    tp = tc - per / (2 * np.pi) * (ee - ecc * np.sin(ee)) # time of periastron

    return tp

@njit
def tp_to_tc(tp, per, ecc, omega):
    """
    Convert Time of Periastron to Time of Transit (time of conjunction).

    Args:
        tp (float): time of periastron
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): argument of peri (radians)
        secondary (bool): calculate time of secondary eclipse instead

    Returns:
        float: time of inferior conjunction (time of transit if system is transiting)
    """
    
    # If ecc >= 1, no tc exists.
    if ecc >= 1:
        return tp

    f = np.pi/2 - omega                                         # true anomaly during transit
    ee = 2 * np.arctan(np.tan( f / 2) * np.sqrt((1 - ecc) / (1 + ecc)))  # eccentric anomaly

    tc = tp + per / (2 * np.pi) * (ee - ecc * np.sin(ee))         # time of conjunction

    return tc