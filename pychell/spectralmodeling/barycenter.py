
# Barycorrpy
from barycorrpy import get_BC_vel
from barycorrpy.utc_tdb import JDUTC_to_BJDTDB

def compute_barycenter_corrections(jdmid, star_name, obsname):
    
    # BJD
    bjd = JDUTC_to_BJDTDB(JDUTC=jdmid, starname=star_name, obsname=obsname, leap_update=True)[0][0]
    
    # bc vel
    bc_vel = get_BC_vel(JDUTC=jdmid, starname=star_name, obsname=obsname, leap_update=True)[0][0]
    
    # Return
    return bjd, bc_vel


#def compute_barycenter_corrections_flux_weighted(t, flux, ...): ...