import optimize.data as optdata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WAVELENGTH_DEFAULTS = {
    'iSHELL': 2350,
    'SPIROU': 2350,
    'TRES': 550,
    'HIRES': 550,
    'HARPS': 550,
    'CSHELL': 2400,
    'NIRSPEC': 2100,
    'CHIRON': 550,
    'CARMENESVIS': 550,
    'PARVI': 1300
}

def get_wavelength(instname):
    try:
        return WAVELENGTH_DEFAULTS[instname]
    except:
        warnings.warn("Wavelength not found.")
        return None

def group_vis_nir(data, cut=1000):
    data_vis = MixedRVData()
    data_nir = MixedRVData()
    for _data in data.values():
        if _data.wavelength < cut:
            data_vis[_data.label] = _data
        else:
            data_nir[_data.label] = _data
    return data_vis, data_nir

class RVData(optdata.Data):
    
    def __init__(self, t, rv, rverr, instname=None, **kwargs):
        super().__init__(t, rv, yerr=rverr, mask=None, label=instname)
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if not hasattr(self, 'wavelength'):
            self.wavelength = get_wavelength(self.instname)
        
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
        
    def __repr__(self):
        return "RVs from " + self.instname
    
    @property
    def time_baseline(self):
        t = self.get_vec('x')
        return np.nanmedian(t)

class MixedRVData(optdata.MixedData):
    
    @property
    def instnames(self):
        return [d.instname for d in self.items()]

    @classmethod
    def from_radvel_file(cls, fname):
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
            data[tel] = RVData(t_all[inds], rv_all[inds], rverr_all[inds], instname=tel)
        return data
    
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
        f = open(fname, 'w')
        f.write('time,mnvel,errvel,tel\n')
        np.savetxt(f, out, fmt='%f,%f,%f,%s')
        f.close()
        
    def make_tel_vec(self):
        tel_vec = np.array([], dtype='<U50')
        t_all = self.get_vec('x', sort=False)
        for instname in self:
            tel_vec = np.concatenate((tel_vec, np.full(len(self[instname].t), fill_value=instname, dtype='<U50')))
        ss = np.argsort(t_all)
        tel_vec = tel_vec[ss]
        return tel_vec
    
    def make_wave_vec(self):
        wave_vec = np.array([], dtype=float)
        x = self.get_vec('x', sort=False)
        ss = np.argsort(x)
        for data in self.values():
            wave_vec = np.concatenate((wave_vec, np.full(data.t.size, fill_value=data.wavelength)))
        wave_vec = wave_vec[ss]
        return wave_vec
    
    def get_inds(self, label):
        tel_vec = self.make_tel_vec()
        inds = np.where(tel_vec == label)[0]
        return inds