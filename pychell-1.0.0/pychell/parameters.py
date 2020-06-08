import numpy as np
from pdb import set_trace as stop

class Parameter:
    
    """A class for a model parameter.

    Attributes:
        name (str): The name of the parameter.
        value (str): The current value of the parameter.
        minv (str): The min bound.
        maxv (str): The max bound.
        vary (str): Whether or not to vary this parameter.
        mcmcscale (str): The mcmc scale step of the parameter.
        commonality (str): What this parameter is common to. Can be anything the user decides to implement.
    """
    
    default_keys_sing = ['name', 'value', 'minv', 'maxv', 'vary', 'mcmcscale', 'commonality']
    default_keys_plur = ['names', 'values', 'minvs', 'maxvs', 'varies', 'mcmcscales', 'commonalities']

    def __init__(self, name=None, value=None, minv=-np.inf, maxv=np.inf, vary=True, mcmcscale=0.5, commonality=None): 
        """Creates a Parameter object.

        Args:
            name (str): The name of the parameter.
            value (str): The value of the parameter.
            minv (str): The min bound.
            maxv (str): The max bound.
            vary (str): Whether or not to vary this parameter.
            mcmcscale (str): The mcmc scale step of the parameter.
            commonality (str): What this parameter is common to. Can be anything the user decides to implement.
        """
        
        self.name = name
        self.value = value
        
        if minv is not None:
            self.minv = minv
        else:
            self.minv = -np.inf
            
        if maxv is not None:
            self.maxv = maxv
        else:
            self.maxv = np.inf
        
        if vary is not None:
            self.vary = vary
        else:
            self.vary = True
            
        if mcmcscale is not None:
            self.mcmcscale = mcmcscale
        else:
            self.mcmcscale = True
            
        if commonality is not None:
            self.commonality = commonality
        else:
            self.commonality = True

    def __repr__(self):
        if self.vary:
            return '(Parameter)  Name: ' + self.name + ' | Value: ' + str(self.value) + ' | Bounds: [' + str(self.minv) + ', ' + str(self.maxv) + ']' + ' | MCMC scale: ' + str(self.mcmcscale) + ' | Commonality:' + str(self.commonality)
        else:
            return '(Parameter)  Name: ' + self.name + ' | Value: ' + str(self.value) + ' (Locked) | Bounds: [' + str(self.minv) + ', ' + str(self.maxv) + ']' + ' | MCMC scale: ' + str(self.mcmcscale) + ' | Commonality:' + str(self.commonality)
    
    def setv(self, **kwargs):
        """Setter method for the attributes.

        kwargs:
            Any available atrribute (str).
        """
        if 'value' in kwargs:
            self.value = kwargs['value']
        if 'minv' in kwargs:
            self.minv = kwargs['minv']
        if 'maxv' in kwargs:
            self.maxv = kwargs['maxv']
        if 'vary' in kwargs:
            self.vary = kwargs['vary']
        if 'mcmcscale' in kwargs:
            self.mcmcscale = kwargs['mcmcscale']
        if 'commonality' in kwargs:
            self.commonality = kwargs['commonality']
        

class Parameters(dict):
    """A container for a set of model parameters which extends the Python 3 dictionary, which is ordered by default.
    """
    
    default_keys_sing = ['name', 'value', 'minv', 'maxv', 'vary', 'mcmcscale', 'commonality']
    default_keys_plur = ['names', 'values', 'minvs', 'maxvs', 'varies', 'mcmcscales', 'commonalities']

    def __init__(self):
        
        # Initiate the actual dictionary.
        super().__init__()
            
            
    @classmethod
    def from_numpy(cls, names, values, minvs=None, maxvs=None, varies=None, mcmcscales=None, commonalities=None):
        """Create a parameters object from numpy arrays
        
        kwargs:
            Iterables of parameter attributes.
        """
        pars = cls()
        n = len(names)
        if minvs is None:
            minvs = [None] * n
        if maxvs is None:
            maxvs = [None] * n
        if varies is None:
            varies = [None] * n
        if mcmcscales is None:
            mcmcscales = [None] * n
        if commonalities is None:
            commonalities = [None] * n
        for i in range(n):
            pars.add_parameter(Parameter(name=names[i], value=values[i], minv=minvs[i], maxv=maxvs[i], vary=varies[i], mcmcscale=mcmcscales[i], commonality=commonalities[i]))
            
        return pars

            
            
    def add_parameter(self, parameter):
        """Adds a parameter to the Parameters dictionary.

        Args:
            parameter (Parameter): The parameter to add.
        """
        self[parameter.name] = parameter
            
    def unpack(self, keys=None):
        """Unpacks values to numpy arrays.

        Args:
            keys (tuple): A tuple of strings containing the keys to unpack, defaults to None for all keys.
            
        Returns:
            vals (dict): A dictionary containing the returned values.
        """
        if keys is None:
            keys_plur = self.default_keys_plur
            keys_sing = self.default_keys_sing
        else:
            if type(keys) is not list:
                keys = [keys]
            keys_plur = keys
            keys_sing = [self.default_keys_sing[self.default_keys_plur.index(keys_plur[i])] for i in range(len(keys_plur))]
        v = {}
        for i in range(len(keys_sing)):
            v[keys_plur[i]] = np.array([getattr(self[pname], keys_sing[i]) for pname in self])
        return v
            
    def pretty_print(self):
        """Prints all parameters and attributes in a readable fashion.
        """
        for key in self.keys():
            print(self[key], flush=True)
            
    
    def setv(self, **kwargs):
        """Setter method for an attribute(s) for all parameters, in order of insertion.

        kwargs:
            Any available Parameter atrribute.
        """
        if 'values' in kwargs:
            for i, pname in enumerate(self):
                self[pname].setv(value=kwargs['values'][i])
        if 'minvs' in kwargs:
            for i, pname in enumerate(self):
                self[pname].setv(minv=kwargs['minvs'][i])
        if 'maxvs' in kwargs:
            for i, pname in enumerate(self):
                self[pname].setv(maxv=kwargs['maxvs'][i])
        if 'varies' in kwargs:
            for i, pname in enumerate(self):
                self[pname].setv(vary=kwargs['varies'][i])
        if 'mcmcscales' in kwargs:
            for i, pname in enumerate(self):
                self[pname].setv(mcmcscale=kwargs['mcmcscales'][i])
        if 'commonalities' in kwargs:
            for i, pname in enumerate(self):
                self[pname].setv(commonality=kwargs['commonalities'][i])
        
    
    def sanity_lock(self):
        """Locks any parameters such that the min value is equal to the max value.
        """
        for pname in self:
            if self[pname].minv == self[pname].maxv:
                self[pname].vary = False
                
                
    def sanity_check(self):
        """Checks for parameters which vary and are out of bounds.
            Returns:
                bad_pars (list): A list containing parameter names (strings) which are out of bounds.
        """
        bad_pars = []
        for pname in self:
            v = self[pname].value
            vary = self[pname].vary
            minv = self[pname].minv
            maxv = self[pname].maxv
            if (v < minv or v > maxv) and vary:
                bad_pars.append(pname)
            
        return bad_pars
    
    
    def num_varied(self):
        nv = 0
        for pname in self:
            nv += int(self[pname].vary)
        return nv
    
    def get_varied(self):
        varied_pars = Parameters()
        for pname in self:
            if self[pname].vary:
                varied_pars.add_parameter(self[pname])
        return varied_pars
    
    def get_locked(self):
        locked_pars = Parameters()
        for pname in self:
            if self[pname].vary:
                locked_pars.add_parameter(self[pname])
        return locked_pars
    
    def get_subspace(self, par_names=None, indices=None):
        sub_pars = Parameters()
        if par_names is not None:
            for pname in par_names:
                sub_pars.add_parameter(self[pname])
        else:
            par_names = list(self.keys())
            for k in indices:
                sub_pars.add_parameter(self[par_names[k]])
        return sub_pars
            
    def index_from_par(self, par_name):
        return list(self.keys()).index(par_name)
    
    def par_from_index(self, k):
        return self[list(self.keys())[k]]
    
    @classmethod
    def oflength(cls, n):
        """Returns a parameters object of some length with dummy names.
        
        Args:
            n (int): The number of parameters to add.
        Returns:
            bad_pars (list): A list containing parameter names (strings) which are out of bounds.
        """
        pars = cls()
        for i in range(n):
            pars.add_parameter(name='par' + str(i + 1), value=0)
        return pars