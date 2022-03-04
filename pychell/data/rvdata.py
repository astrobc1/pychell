# Maths
import numpy as np
import pandas as pd


class RVData:
    
    __slots__ = ['t', 'rv', 'rverr', 'instname', 'wavelength']
    
    def __init__(self, t, rv, rverr, instname=None, wavelength=None):
        """Construct an RV data object for a particular instrument.

        Args:
            t (np.ndarray): The time vector.
            rv (np.ndarray): The RVs vector.
            rverr (np.ndarray): The RVs error vector.
            instname (np.ndarray, optional): The label for this dataset. Defaults to None initially, and is then replaced by the the key provided when constructing a composite dataset.
            wavelength (np.ndarray, optional): The effective wavelength of the dataset. Defaults to None.
        """
        self.t = t
        self.rv = rv
        self.rverr = rverr
        self.instname = instname
        self.wavelength = wavelength

    def __repr__(self):
        return f"{len(self.t)} RVs from {self.instname}"

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
        t, rv, rverr = np.loadtxt(fname, delimiter=delimiter, usecols=usecols, unpack=True, skiprows=skiprows)
        data = cls(t, rv, rverr, instname=instname, wavelength=wavelength)
        return data


class CompositeRVData(dict):

    def __init__(self):
        super().__init__()
        self.indices = {}

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
        return self.gen_tel_vec()
 
    def get_view(self, instnames):
        """Returns a view into sub data objects.
        Args:
            labels (str or list of strings): The labels to get.
        Returns:
            type(self): A view into the original data object.
        """
        data_view = self.__class__()
        instnames = np.atleast_1d(instnames)
        for instname in instnames:
            data_view[instname] = self[instname]
        return data_view
    
    @property
    def time_baseline(self):
        """T_max - T_min

        Returns:
            float: The time baseline.
        """
        t = self.t
        return np.nanmax(t) - np.nanmin(t)

    
    def gen_tel_vec(self):
        """Generates a vector where each index corresponds to the label of measurement x, sorted by x as well.

        Returns:
            np.ndarray: The label vector sorted according to self.indices.
        """
        tel_vec = np.empty(len(self.t), dtype='<U50')
        for data in self.values():
            tel_vec[self.indices[data.instname]] = data.instname
        return tel_vec

    def get_vec(self, attr, sort=True, labels=None):
        """Gets a vector for certain labels and possibly sorts it.

        Args:
            attr (str): The attribute to get.
            sort (bool): Whether or not to sort the returned vector according to x.
            labels (list of strings, optional): The labels to get. Defaults to all.

        Returns:
            np.ndarray: The vector, sorted according to x if sort=True.
        """
        n = np.sum([data.t.size for data in self.values()])
        if labels is None:
            labels = list(self.keys())
        if sort:
            out = np.zeros(n)
            for label in labels:
                out[self.indices[label]] = getattr(self[label], attr)
        else:
            out = np.array([], dtype=float)
            for label in labels:
                out = np.concatenate((out, getattr(self[label], attr)))
        return out
    
    def gen_indices(self):
        """Utility function to generate the indices of each dataset (when sorted according to x).

        Returns:
            dict: A dictionary with keys = data labels, values = numpy array of indices (ints).
        """
        indices = {}
        tel_vec = np.array([], dtype="<U50")
        t = np.array([], dtype=float)
        for data in self.values():
            t = np.concatenate((t, data.t))
            tel_vec = np.concatenate((tel_vec, np.full(len(data.t), fill_value=data.instname, dtype="<U50")))
        ss = np.argsort(t)
        tel_vec = tel_vec[ss]
        for data in self.values():
            inds = np.where(tel_vec == data.instname)[0]
            indices[data.instname] = inds
        return indices
    
    def __setitem__(self, instname, data):
        if data.instname is None:
            data.instname = instname
        super().__setitem__(instname, data)
        self.indices = self.gen_indices()

    @property
    def t(self):
        return self.get_vec("t")

    @property
    def rv(self):
        return self.get_vec("rv")

    @property
    def rverr(self):
        return self.get_vec("rverr")
    
    def __delitem__(self, key):
        super().__delitem__(key)
        self.indices = self.gen_indices()

    def __repr__(self):
        s = ""
        for _data in self.values():
            s += f"   {_data}\n"
        s = s[0:-1]
        return s