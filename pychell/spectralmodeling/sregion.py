import numpy as np

class SpecRegion1d:
    
    __slots__ = ['pixmin', 'pixmax', 'wavemin', 'wavemax', '_label', 'order', 'fiber']
    
    def __init__(self, pixmin=None, pixmax=None, wavemin=None, wavemax=None, label=None, order=None, fiber=None):
        """Initiate a spectral region.

        Args:
            pixmin (int): The min pixel.
            pixmax (int): The max pixel.
            wavemin (float): The minimum wavelength
            wavemax (float): The maximum wavelength
            label (str, optional): A custom label for this region. Defaults to Order#.
            order (int, optional): The echelle order number, if relevant. Defaults to None.
            fiber (int, optional): The fiber number.
        """
        self.pixmin = pixmin
        self.pixmax = pixmax
        self.wavemin = wavemin
        self.wavemax = wavemax
        self.order = order
        self._label = label
        self.fiber = fiber
        
    def __len__(self):
        return self.pix_len()
    
    def wave_len(self):
        return self.wavemax - self.wavemin
    
    def pix_len(self):
        return self.pixmax - self.pixmin
        
    def pix_within(self, pixels, pad=0):
        good = np.where((pixels >= self.pixmin - pad) & (pixels <= self.pixmax + pad))[0]
        return good
        
    def wave_within(self, waves, pad=0):
        good = np.where((waves >= self.wavemin - pad) & (waves <= self.wavemax + pad))[0]
        return good
    
    def midwave(self):
        return self.wavemin + self.wave_len() / 2
    
    def midpix(self):
        return self.pixmin + self.pix_len() / 2
        
    def pix_per_wave(self):
        return (self.pixmax - self.pixmin) / (self.wavemax - self.wavemin)
    
    def wave_per_pix(self):
        return  (self.wavemax - self.wavemin) / (self.pixmax - self.pixmin)

    @property
    def label(self):
        if self._label is not None:
            return str(self._label)
        else:
            return f"Order{self.order}"

    
    def __repr__(self):
        return f"{self.label}: Pixels = {self.pixmin} - {self.pixmax}, Wavelength = {round(self.wavemin, 3)} - {round(self.wavemax, 3)} nm"