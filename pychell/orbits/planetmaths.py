import numpy as np
from numba import njit, jit
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

def compute_mass(per, ecc, k, mstar):
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

def compute_sa(per, mstar):
    return (G_MKS / (4 * np.pi**2))**(1 / 3) * (mstar * MSUN_KG)**(1 / 3) * (per * 86400)**(2 / 3) / AU_M

def compute_sa_deriv_mstar(per, mstar):
    return (G_MKS / (4 * np.pi**2))**(1 / 3) * (mstar * MSUN_KG)**(-2 / 3) / 3 * (per * 86400)**(2 / 3) * (MSUN_KG / AU_M)

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

def bin_phased_rvs(phases, rvs, unc, nbins=10):
    """Bins the phased RVs.

    Args:
        phases (np.ndarray): The phases in [0, 1].
        rvs (np.ndarray): The data rvs.
        unc (np.ndarray): The corresponding data uncertainties.
        window (float): The bin size.

    Returns:
        np.ndarray: The binned phases.
        np.ndarray: The binned RVs.
        np.ndarray: The binned uncertainties.
    """
    
    binned_phases = np.full(nbins, np.nan)
    binned_rvs = np.full(nbins, np.nan)
    binned_unc = np.full(nbins, np.nan)
    bins = np.linspace(0, 1, num=nbins+1)
    for i in range(nbins):
        inds = np.where((phases >= bins[i]) & (phases < bins[i + 1]))[0]
        n = len(inds)
        if n == 0:
            continue
        w = 1 / unc[inds]**2
        w /= np.nansum(w)
        binned_phases[i] = np.nanmean(phases[inds])
        binned_rvs[i] = pcmath.weighted_mean(rvs[inds], w)
        binned_unc[i] = pcmath.weighted_stddev(rvs[inds], w) / np.sqrt(n)

    return binned_phases, binned_rvs, binned_unc

def compute_mass_deriv_mstar(per, ecc, k, mstar):
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

def compute_density(mplanet, rplanet):
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

def compute_density_deriv_rplanet(rplanet, mplanet):
    """A helper function that computes (d rho)/(d rplanet) in useful units given values for mass and radius of the planet.

    Args:
        mplanet (float): The mass of the planet in Earth units.
        rplanet (float): The radius of the planet in Earth units.

    Returns:
        float: The derivative (d rho)/(d mplanet) evaluated at (mplanet, rplanet) in units of cgs / cm.
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

@njit(nogil=True)
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
    ta = true_anomaly(t, tp, per, ecc)
    rv = k * (np.cos(ta + w) + ecc * np.cos(w))

    # Return rv
    return rv