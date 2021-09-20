# Optimize deps
from optimize.data import SimpleSeries, CompositeSimpleSeries

# Maths
import numpy as np
import pandas as pd


class RVData(SimpleSeries):
    
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

    def __repr__(self):
        return f"{len(self.t)} RVs from {self.instname}"

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
        """T_max - T_min

        Returns:
            float: The time baseline.
        """
        return np.nanmax(self.t) - np.nanmin(self.t)

    @classmethod
    def from_file(cls, fname, instname=None, delimiter=",", skiprows=0, usecols=None, wavelength=None):
        """Constructor from a delimited file.

        Args:
            fname (str): The full path and filename.
            instname (str, optional): The name of the instrument. Defaults to None.
            delimiter (str, optional): The delimiter. Defaults to ",".
            skiprows (int, optional): Which rows to skip. Defaults to 0.
            usecols (tuple of ints, optional): The columns corresponding to time, rv, rverr. Defaults to (0, 1, 2).
            wavelength (float, optional): The wavelength of this dataset. Defaults to None.

        Returns:
            RVData: The RV data object for this instrument.
        """
        if usecols is None:
            usecols = (0, 1, 2)
        t, rvs, rvs_unc = np.loadtxt(fname, delimiter=delimiter, usecols=usecols, unpack=True, skiprows=skiprows)
        data = cls(t, rvs, rvs_unc, instname=instname, wavelength=wavelength)
        return data


class CompositeRVData(CompositeSimpleSeries):

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
    
    def gen_wave_vec(self):
        wave_vec = np.array([], dtype=float)
        t = np.array([], dtype=float)
        for data in self.values():
            t = np.concatenate((t, data.t))
            wave_vec = np.concatenate((wave_vec, np.full(data.t.size, fill_value=data.wavelength)))
        ss = np.argsort(t)
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
        tel_vec = self.gen_instname_vec()
        out = np.array([times, rvs, rvserr, tel_vec], dtype=object).T
        with open(fname, 'w') as f:
            f.write('time,mnvel,errvel,tel\n')
            np.savetxt(f, out, fmt='%f,%f,%f,%s')
        
    def gen_instname_vec(self):
        return self.gen_label_vec()
 
    def get_view(self, instnames):
        """Returns a view into sub data objects. Really just a forward method, propogating instnames -> labels.

        Args:
            instnames (list): A list of instruments as strings.

        Returns:
            CompositeData: A view into the original data object (not a copy).
        """
        return super().get_view(labels=instnames)
    
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
    def time_baseline(self):
        """T_max - T_min

        Returns:
            float: The time baseline.
        """
        t = self.t
        return np.nanmax(t) - np.nanmin(t)