// Import headers with #include
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


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