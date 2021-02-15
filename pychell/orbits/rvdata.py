import optimize.data as optdata
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

class RVData(optdata.Data):
    
    __slots__ = ['x', 'y', 'yerr', 'mask', 'label', 'wavelength']
    
    def __init__(self, t, rv, rverr, instname=None, wavelength=None):
        super().__init__(t, rv, yerr=rverr, label=instname)
        self.wavelength = wavelength
        
    @property
    def t(self):
        return self.x
    
    @property
    def rv(self):
        return self.y
    
    @property
    def rverr(self):
        return self.yerr
    
    @property
    def instname(self):
        return self.label
    
    @property
    def time_baseline(self):
        t = self.get_vec('x')
        return np.max(t) - np.min(t)
        
    def __repr__(self):
        return str(len(self.t)) + " RVs from " + self.instname
    
    
class CompositeRVData(optdata.CompositeData):
    
    @property
    def instnames(self):
        return [d.instname for d in self.items()]

    @classmethod
    def from_radvel_file(cls, fname, wavelengths=None):
        """Constructs a new RV data object from a standard radvel csv file.

        Args:
            fname (str): The full path to the file.

        Returns:
            MixedRVData: The MixedRVData set.
        """
        data = cls()
        rvdf = pd.read_csv(fname, sep=',', comment='#')
        tel_vec_unq = rvdf.tel.unique()
        tel_vec = rvdf.tel.to_numpy().astype('<U50')
        t_all = rvdf.time.to_numpy()
        rv_all = rvdf.mnvel.to_numpy()
        rverr_all = rvdf.errvel.to_numpy()
        for tel in tel_vec_unq:
            inds = np.where(tel_vec == tel)[0]
            if wavelengths is not None:
                wavelength = wavelengths[tel]
            else:
                wavelength = None
            data[tel] = RVData(t_all[inds], rv_all[inds], rverr_all[inds], instname=tel, wavelength=wavelength)
        return data
    
    def get_vec(self, key, labels=None, sort=True):
        """Combines a certain vector from all labels into one array, and can then sort it according to time.

        Args:
            key (str): The key to get (t, rv, rverr)
            labels (list): A list of labels (dict keys).

        Returns:
            np.ndarray: The vector, sorted according to x.
        """
        if labels is None:
            labels = list(self.keys())
        out = np.array([], dtype=float)
        if sort:
            x = np.array([], dtype=float)
        for label in labels:
            out = np.concatenate((out, getattr(self[label], key)))
            if sort:
                x = np.concatenate((x, self[label].t))
            
        # Sort
        if sort:
            ss = np.argsort(x)
            out = out[ss]

        return out
    
    def get_wave_vec(self):
        wave_vec = np.array([], dtype=float)
        t = self.get_vec('t', sort=False)
        ss = np.argsort(t)
        for data in self.values():
            wave_vec = np.concatenate((wave_vec, np.full(data.t.size, fill_value=data.wavelength)))
        wave_vec = wave_vec[ss]
        return wave_vec
    
    def to_radvel_file(self, fname):
        """Creates a radvel input file from an instance.

        Args:
            fname (str): The full path and filename of the file to create. If the file exists, it is overwritten.
        """
        times = self.get_vec('t', sort=True)
        rvs = self.get_vec('rv', sort=True)
        rvserr = self.get_vec('rverr', sort=True)
        tel_vec = self.make_tel_vec()
        out = np.array([times, rvs, rvserr, tel_vec], dtype=object).T
        with open(fname, 'w') as f:
            f.write('time,mnvel,errvel,tel\n')
            np.savetxt(f, out, fmt='%f,%f,%f,%s')
        
    def make_tel_vec(self):
        tel_vec = np.array([], dtype='<U50')
        t_all = self.get_vec('t', sort=False)
        for instname in self:
            tel_vec = np.concatenate((tel_vec, np.full(len(self[instname].t), fill_value=instname, dtype='<U50')))
        ss = np.argsort(t_all)
        tel_vec = tel_vec[ss]
        return tel_vec
    
    def get_inds(self, label):
        tel_vec = self.make_tel_vec()
        inds = np.where(tel_vec == label)[0]
        return inds
    
    def get_instruments(self, labels):
        data_out = MixedRVData()
        for label in labels:
            data_out[label] = self[label]
        return data_out
    
    def get(self, instnames):
        """Returns a view into sub data objects.

        Args:
            instnames (list): A list of instnmes (str).

        Returns:
            CompositeData: A view into the original data object.
        """
        data_view = self.__class__()
        for label in instnames:
            data_view[label] = self[label]
        return data_view
 
def group_vis_nir(data, cut=1000):
    """Groups vis and nir data into two different dicts.

    Args:
        data (MixedRVData): The RV data.
        cut (float, optional): The cut between vis and nir in nm. Defaults to 1000 nm.

    Returns:
        MixedRVData: The vis data.
        MixedRVData: The nir data.
    """
    data_vis = MixedRVData()
    data_nir = MixedRVData()
    for _data in data.values():
        if _data.wavelength < cut:
            data_vis[_data.label] = _data
        else:
            data_nir[_data.label] = _data
    return data_vis, data_nir
