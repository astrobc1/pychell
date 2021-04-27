import numpy as np
from numba import njit
import pychell.maths as pcmath
import scipy.constants

# CONSTANTS
MASS_JUPITER_EARTH_UNITS = 317.82838 # mass of jupiter in earth masses
MASS_EARTH_GRAMS = 5.972181578208704E27 # mass of earth in grams
RADIUS_EARTH_CM = 6.371009E8 # radius of earth in cm
YEAR_EARTH_DAYS = 365.25 # one year for earth in days
K_JUPITER_P_EARTH = 28.4329 # the semi-amplitude of jupiter for a one year orbit
MSUN_KG = 1.988435E30 # The mass of the sun in kg
G_MKS = scipy.constants.G # The Newtonian gravitational constant in mks units
AU_M = 1.496E11 # 1 AU in meters

def compute_planet_mass(per, ecc, k, mstar):
    """Computes the planet mass from the semi-amplitude equation.

    Args:
        per (float): The period of the orbit in days.
        ecc (float): The eccentricity of the orbit.
        k (float): The RV semi-amplitude in m/s.
        mstar (float): The mass of the star in solar units.

    Returns:
        float: The planet mass in units of Earth masses.
    """
    return k * np.sqrt(1 - ecc**2) / K_JUPITER_P_EARTH * (per / YEAR_EARTH_DAYS)**(1 / 3) * mstar**(2 / 3) * MASS_JUPITER_EARTH_UNITS

def get_phases(t, per, tc):
    """Given input times, a period, and time of conjunction, returns the phase [0, 1] at each time t, in the same order as t, so that the phases returned will likely be unsorted, but will match up with the other vectors.
    
    Args:
        t (np.ndarray): The times.
        per (float): The period of the planet.
        tc (float): The time of conjunction (time of transit).

    Returns:
        np.ndarray: The phases between 0 and 1
    """
    phases = ((t - tc - per / 2) % per) / per
    return phases

def bin_phased_rvs(phases, rvs, unc, window=0.1):
    """Bins the phased RVs.

    Args:
        phases (np.ndarray): The phases, [0, 1).
        rvs (np.ndarray): The data rvs.
        unc (np.ndarray): The corresponding data uncertainties.
        window (float): The bin size.

    Returns:
        np.ndarray: The binned phases.
        np.ndarray: The binned RVs.
        np.ndarray: The binned uncertainties.
    """
    
    binned_phases = []
    binned_rvs = []
    binned_unc = []
    i = 0
    while i < len(phases):
        inds = np.where((phases >= phases[i]) & (phases < phases[i] + window))[0]
        n = len(inds)
        w = 1 / unc[inds]**2
        w /= np.nansum(w)
        binned_phases.append(np.mean(phases[inds])) # unlike radvel, just use unweighted mean for time.
        binned_rvs.append(np.sum(w * rvs[inds]))
        binned_unc.append(1 / np.sqrt(np.sum(1 / unc[inds]**2)))
        i += n

    return binned_phases, binned_rvs, binned_unc

def compute_planet_mass_deriv_mstar(per, ecc, k, mstar):
    """Computes the derivative of the semi-amplitude equation inverted for mass, d(M_planet) / d(M_Star)

    Args:
        per (float): The period of the orbit in days.
        ecc (float): The eccentricity.
        k (float): The RV semi-amplitude in m/s.
        mstar (float): The mass of the star in solar units.

    Returns:
        float: The derivative (unitless).
    """
    a = k * np.sqrt(1 - ecc**2) / K_JUPITER_P_EARTH * (per / YEAR_EARTH_DAYS)**(1 / 3) * MASS_JUPITER_EARTH_UNITS
    dMp_dMstar = (2 / 3) * a * mstar**(-1 / 3)
    return dMp_dMstar

def compute_planet_density(mplanet, rplanet):
    """Computes the planet density.

    Args:
        mplanet (float): The mass of the planet in earth units.
        rplanet (float): The radius of the planet in earth units.

    Returns:
        float: The density of the planet in cgs units.
    """
    mplanet_grams = mplanet * MASS_EARTH_GRAMS
    rplanet_cm = rplanet * RADIUS_EARTH_CM
    rho_cgs = (3 * mplanet_grams) / (4 * np.pi * rplanet_cm**3)
    return rho_cgs

def compute_planet_density_deriv_rplanet(mplanet, rplanet):
    """A helper function that computes (d rho)/(d rplanet) in useful units given values for mass and radius of the planet.

    Args:
        mplanet (float): The mass of the planet in Earth units.
        rplanet (float): The radius of the planet in Earth units.

    Returns:
        float: The derivative (d rho)/(d mplanet) evaluated at (mplanet, rplanet).
    """
    mplanet_grams = mplanet * MASS_EARTH_GRAMS
    rplanet_cm = rplanet * RADIUS_EARTH_CM
    d_rho_d_rplanet = (9 * mplanet_grams) / (4 * np.pi * rplanet_cm**4)
    return d_rho_d_rplanet

def ffprime_spots(time, flux, flux_unc, cvbs=True, rstar=1.0, f=0.1, sampling=0.5):
    """Presidcts the spot induced activity RV signature via the F*F' method using https://arxiv.org/pdf/1110.1034.pdf.

    Args:
        time (np.ndarray): The times for the light curve.
        flux (np.ndarray): The flux of the light curve.
        rstar (float): The radius of the star in solar units.
        f (float): The relative flux drop for a spot at the disk center.
        sampling(float): The spacing for the knots in units of the time array.
        
    Returns:
        np.ndarray: The predicted RV signature from stellar spots.
    """
    
    knots = np.arange(time[0], time[-1], sampling)
    weights = 1 / flux_unc**2
    weights /= np.nansum(weights)
    cspline = pcmath.cspline_fit(time, flux, knots=knots, weights=weights)
    rv_pred_spots = -1.0 * cspline(time) * cspline(time, 1) * rstar / f
    return rv_pred_spots

@njit(nogil=True)
def solve_kepler_all_times(mas, ecc):
    eas = np.zeros_like(mas)
    for i in range(mas.size):
        eas[i] = _solve_kepler(mas[i], ecc)
    return eas

@njit(nogil=True)
def _solve_kepler(ma, ecc):
    """Solve Kepler's equation for one planet and one time. This code is nearly identical to the RadVel implemenation (BJ Fulton et al. 2018). Kepler's equation is solved using a higher order Newton's method. Note that for RV modeling, solving Kepler's eq. is typically not a bottleneck.
    
    Args:
        ma (float): mean anomaly.
        eccarr (float): eccentricity.
        
    Returns:
        float: The eccentric anomaly.
    """

    # Convergence criterion
    conv = 1E-10
    k = 0.85
    max_iters = 200
    
    # First guess for ea
    ea = ma + np.sign(np.sin(ma)) * k * ecc
    fi = ea - ecc * np.sin(ea) - ma
    
    # Counter
    count = 0
    
    # Break when converged
    while True and count < max_iters:
        
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

@njit(nogil=True)
def true_anomaly(t, tp, per, ecc):
    """
    Calculate the true anomaly for a given time, period, eccentricity. This requires solving Kepler's equation.

    Args:
        t (np.ndarray): The times.
        tp (float): The time of periastron.
        per (float): The period of the orbit in units of t.
        ecc (float): The eccentricity of the bounded orbit.

    Returns:
        np.ndarray: true anomoly at each time
    """
    
    m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
    ea = solve_kepler_all_times(m, ecc)
    n1 = 1.0 + ecc
    n2 = 1.0 - ecc
    ta = 2.0 * np.arctan((n1 / n2)**0.5 * np.tan(ea / 2.0))
    return ta

@njit(nogil=True)
def tc_to_tp(tc, per, ecc, w):
    """
    Convert Time of Transit (time of conjunction) to Time of Periastron Passage.

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
    ee = 2 * np.arctan(np.tan(f / 2) * np.sqrt((1 - ecc) / (1 + ecc)))
    tp = tc - per / (2 * np.pi) * (ee - ecc * np.sin(ee))

    return tp

@njit(nogil=True)
def tp_to_tc(tp, per, ecc, w):
    
    """
    Convert Time of Periastron to Time of Transit (time of conjunction).

    Args:
        tp (float): time of periastron
        per (float): period [days]
        ecc (float): eccentricity
        w (float): argument of periastron (radians).

    Returns:
        float: The time of conjunction.
    """
    
    # If ecc >= 1, no tc exists.
    if ecc >= 1:
        return tp

    f = np.pi / 2 - w                                         # true anomaly during transit
    ee = 2 * np.arctan(np.tan( f / 2) * np.sqrt((1 - ecc) / (1 + ecc)))  # eccentric anomaly

    tc = tp + per / (2 * np.pi) * (ee - ecc * np.sin(ee))         # time of conjunction

    return tc

def compute_semimajor_axis(mcmc_result, mstar=None, mstar_unc=None):
    """Computes the semi-major axis of each planet.

    Args:
        mcmc_result (dict): The returned value from calling sample.
        mstar (float): The mass of the star in solar units.
        mstar (list): The uncertainty of the mass of the star in solar units, lower, upper.

    Returns:
        (dict): The semi-major axis, lower, and upper uncertainty of each planet in a dictionary.
    """
    if mstar is None:
        mstar = self.mstar
    if mstar_unc is None:
        mstar_unc = self.mstar_unc
    aplanets = {} # In AU
    for planet_index in self.planets_dict:
        perdist = []
        tpdist = []
        eccdist = []
        wdist = []
        kdist = []
        adist = []
        pars = copy.deepcopy(mcmc_result["pmed"])
        for i in range(mcmc_result["n_steps"]):
            for pname in self.planets_dict[planet_index]["basis"].pnames:
                if pars[pname].vary:
                    ii = pars.index_from_par(pname, rel_vary=True)
                    pars[pname].value = mcmc_result["chains"][i, ii]
            per, tp, ecc, w, k = self.planets_dict[planet_index]["basis"].to_standard(pars)
            perdist.append(per)
            tpdist.append(tp)
            eccdist.append(ecc)
            wdist.append(w)
            kdist.append(k)
            a = (G_MKS / (4 * np.pi**2))**(1 / 3) * (mstar * MSUN_KG)**(1 / 3) * (per * 86400)**(2 / 3) / AU_M
            adist.append(a)
        val, unc_low, unc_high = self.sampler.chain_uncertainty(adist)
        if self.mstar_unc is not None:
            da_dMstar = (G_MKS / (4 * np.pi**2))**(1 / 3) * (mstar * MSUN_KG)**(-2 / 3) / 3 * (per * 86400)**(2 / 3) * (MSUN_KG / AU_M) # in AU / M_SUN
            unc_low = np.sqrt(unc_low**2 + da_dMstar**2 * mstar_unc[0]**2)
            unc_high = np.sqrt(unc_high**2 + da_dMstar**2 * mstar_unc[1]**2)
            aplanets[planet_index] = (val, unc_low, unc_high)
        else:
            aplanets[planet_index] = (val, unc_low, unc_high)
    return aplanets

def compute_planet_masses(self, mcmc_result):
    """Computes the value of msini and uncertainty for each planet in units of Earth Masses.

    Args:
        mcmc_result (dict): The returned value from calling sample.

    Returns:
        (dict): The mass, lower, and upper uncertainty of each planet in a dictionary.
    """
    mstar = self.mstar
    mstar_unc = self.mstar_unc
    msiniplanets = {} # In earth masses
    for planet_index in self.planets_dict:
        perdist = []
        tpdist = []
        eccdist = []
        wdist = []
        kdist = []
        mdist = []
        pars = copy.deepcopy(mcmc_result["pmed"])
        for i in range(mcmc_result["n_steps"]):
            for pname in self.planets_dict[planet_index]["basis"].pnames:
                if pars[pname].vary:
                    ii = pars.index_from_par(pname, rel_vary=True)
                    pars[pname].value = mcmc_result["chains"][i, ii]
            per, tp, ecc, w, k = self.planets_dict[planet_index]["basis"].to_standard(pars)
            perdist.append(per)
            tpdist.append(tp)
            eccdist.append(ecc)
            wdist.append(w)
            kdist.append(k)
            mdist.append(compute_planet_mass(per, ecc, k, mstar))
        val, unc_low, unc_high = self.sampler.chain_uncertainty(mdist)
        if self.mstar_unc is not None:
            unc_low = np.sqrt(unc_low**2 + compute_planet_mass_deriv_mstar(per, ecc, k, mstar)**2 * mstar_unc[0]**2)
            unc_high = np.sqrt(unc_high**2 + compute_planet_mass_deriv_mstar(per, ecc, k, mstar)**2 * mstar_unc[1]**2)
            msiniplanets[planet_index] = (val, unc_low, unc_high)
        else:
            msiniplanets[planet_index] = (val, unc_low, unc_high)
    return msiniplanets

def compute_planet_densities(self, mcmc_result):
    """Computes the value of msini and uncertainty for each planet in units of Earth Masses.

    Args:
        mcmc_result (dict): The returned value from calling sample.
    Returns:
        (dict): The density, lower, and upper uncertainty of each planet in a dictionary, in units of grams/cm^3.
    """
    mstar = self.mstar
    mstar_unc = self.mstar_unc
    rplanets = self.rplanets
    mplanets = self.compute_planet_masses(mcmc_result)
    rhoplanets = {} # In jupiter masses
    for planet_index in self.planets_dict:
        perdist = []
        tpdist = []
        eccdist = []
        wdist = []
        kdist = []
        mdist = []
        rhodist = []
        pars = copy.deepcopy(mcmc_result["pmed"])
        for i in range(mcmc_result["n_steps"]):
            for pname in self.planets_dict[planet_index]["basis"].pnames:
                if pars[pname].vary:
                    ii = pars.index_from_par(pname, rel_vary=True)
                    pars[pname].value = mcmc_result["chains"][i, ii]
            per, tp, ecc, w, k = self.planets_dict[planet_index]["basis"].to_standard(pars)
            perdist.append(per)
            tpdist.append(tp)
            eccdist.append(ecc)
            wdist.append(w)
            kdist.append(k)
            mplanet = compute_planet_mass(per, ecc, k, mstar)
            rplanet_val = rplanets[planet_index][0]
            rplanet_unc_low = rplanets[planet_index][1]
            rplanet_unc_high = rplanets[planet_index][2]
            rhodist.append(compute_planet_density(mplanet, rplanet_val))
        val, unc_low, unc_high = self.sampler.chain_uncertainty(rhodist)
        if rplanets[planet_index] is not None:
            mplanet = mplanets[planet_index][0]
            unc_low = np.sqrt(unc_low**2 + compute_planet_density_deriv_rplanet(rplanet_val, mplanet)**2 * rplanet_unc_low**2)
            unc_high = np.sqrt(unc_high**2 + compute_planet_density_deriv_rplanet(rplanet_val, mplanet)**2 * rplanet_unc_high**2)
            rhoplanets[planet_index] = (val, unc_low, unc_high)
        else:
            rhoplanets[planet_index] = (val, unc_low, unc_high)
    return rhoplanets
