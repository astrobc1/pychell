// Import headers with #include
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/*Solve Kepler's Equation for one planet.
    Args:
        ma (double): mean anomaly
        ecc (double): eccentricity
    Returns:
        float: The eccentric anomaly.
*/
double solve_kepler(double ma, double ecc) {

    // Convergence criterion
    double conv = 1.0E-12;
    double k = 0.85;

    // init ea and fi
    double ea, fi;

    // Init guess
    double sinma = sin(ma);
    double sign;
    if (sinma < 0) {
        sign = 1.0;
    } else if (sinma > 0) {
        sign = -1.0;
    } else {
        sign = 0.0;
    }

    ea = ma + sign * k * ecc;
    fi = ea - ecc * sin(ea) - ma
    
    // Counter
    long count = 0;

    // Init remaining vars
    double fip, fipp, d1, ea_new;

    // Break when converged
    while true {

        // Increase counter
        count += 1;

        // Update ea
        fip = 1 - ecc * cos(ea)
        fipp = ecc * sin(ea)
        fippp = 1 - fip
        d1 = -1 * fi / fip
        d1 = -1 * fi / (fip + d1 * fipp / 2.0)
        d1 = -1 * fi / (fip + d1 * fipp / 2.0 + d1 * d1 * fippp / 6.0)
        ea_new = ea + d1

        // Check convergence
        fi = ea_new - ecc * sin(ea_new) - ma;
        if fi < conv {
            break
        }
        ea = ea_new

        return ea_new

    }

}