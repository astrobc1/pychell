import numpy as np
import pychell.orbits.maths as planetmath

class OrbitBasis:
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
        raise NotImplementedError(f"Must implement a from_standard method for class {cls.__name__}")

class StandardOrbitBasis(OrbitBasis):
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

class TCOrbitBasis(OrbitBasis):
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

class TCSQEOrbitBasis(OrbitBasis):
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
                          (sqecosw_unc / (sqecosw_unc**2 + sqesinw_unc**2))**2 * sqesinw_unc**2)
        
        return (per_unc, tp_unc, ecc_unc, w_unc, k_unc)

class TCEOrbitBasis(OrbitBasis):
    
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
                          (sqecosw_unc / (sqecosw_unc**2 + sqesinw_unc**2))**2 * sqesinw_unc**2)
        
        return (per_unc, tp_unc, ecc_unc, w_unc, k_unc)
