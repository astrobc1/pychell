import optimize.data as optdata
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

class RVData(optdata.Data1d):
    
    __slots__ = ['x', 'y', 'yerr', 'mask', 'label', 'wavelength']
    
    def __init__(self, t, rv, rverr, instname=None, wavelength=None):
        """Construct an RV data object for a particular instrument.

        Args:
            t (np.ndarray): The time vector.
            rv (np.ndarray): The RVs vector.
            rverr (np.ndarray): The RVs error vector.
            instname (np.ndarray, optional): The label for this dataset. Defaults to None initially, and is then replaced by the the key provided when constructing a composite dataset.
            wavelength (np.ndarray, optional): The effective wavelength of the dataset. Defaults to None.
        """
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
    
    @classmethod
    def from_file(cls, fname, instname=None, delimiter=",", skiprows=0, usecols=None, wavelength=None):
        if usecols is None:
            usecols = (0, 1, 2)
        t, rvs, rvs_unc = np.loadtxt(fname, delimiter=delimiter, usecols=usecols, unpack=True, skiprows=skiprows)
        data = cls(t, rvs, rvs_unc, instname=instname, wavelength=wavelength)
        return data
        
    def __repr__(self):
        return str(len(self.t)) + " RVs from " + self.instname
    
    
class CompositeRVData(optdata.CompositeData1d):
    
    @property
    def instnames(self):
        return np.array(list(self.keys()), dtype="<U50")

    @classmethod
    def from_radvel_file(cls, fname, wavelengths=None):
        """Constructs a new RV data object from a standard radvel csv file.

        Args:
            fname (str): The full path to the file.

        Returns:
            CompositeRVData: The CompositeRVData set.
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
        return self.make_label_vec()
    
    def get_inds(self, label):
        tel_vec = self.make_tel_vec()
        inds = np.where(tel_vec == label)[0]
        return inds
    
    def get(self, instnames):
        """Returns a view into sub data objects.

        Args:
            instnames (list): A list of instnmes (str).

        Returns:
            CompositeData: A view into the original data object.
        """
        return super().get(labels=instnames)
    
    @property
    def tel_vec(self):
        return self.label_vec