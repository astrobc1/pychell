import numpy as np

class SpecRegion2d:
    
    __slots__ = ['pixmin', 'pixmax', 'orderbottom', 'ordertop', 'poly_bottom', 'poly_top']
    
    def __init__(self, pixmin=None, pixmax=None, orderbottom=None, ordertop=None, poly_bottom=None, poly_top=None):
        """Initiate a spectral region.

        Args:
            pixmin (int): The min pixel for each order.
            pixmax (int): The max pixel for each order.
            orderbottom (int): The bottom order number.
            orderotop (int): The top order number.
            poly_bottom (np.ndarray): The polynomial to mask the bottom of the image with.
            poly_bottom (np.ndarray): The polynomial to mask the top of the image with.
        """
        self.pixmin = pixmin
        self.pixmax = pixmax
        self.orderbottom = orderbottom
        self.ordertop = ordertop
        self.poly_bottom = poly_bottom
        self.poly_top = poly_top
    
    def __repr__(self):
        return f"Echellogram Region: Pixels = {self.pixmin} - {self.pixmax}, Orders = {self.ordermin} - {self.ordermax}"

    def mask_image(self, image):

        # dims
        ny, nx = image.shape

        # Mask left/right
        image[:, 0:self.pixmin] = np.nan
        image[:, self.pixmax+1:] = np.nan

        # Top polynomial
        xarr = np.arange(nx)
        ybottom = np.polyval(self.poly_bottom, xarr)
        ytop = np.polyval(self.poly_top, xarr)

        # Mask
        yarr = np.arange(ny)
        for x in range(nx):
            bad = np.where((yarr < ybottom[x]) | (yarr > ytop[x]))[0]
            image[bad, x] = np.nan

    @property
    def ordermin(self):
        return np.min([self.orderbottom, self.ordertop])

    @property
    def ordermax(self):
        return np.max([self.orderbottom, self.ordertop])

    @property
    def n_orders(self):
        return self.ordermax - self.ordermin + 1