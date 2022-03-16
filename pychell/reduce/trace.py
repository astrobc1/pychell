# Maths
import numpy as np
import scipy.signal

# pychell
import pychell.maths as pcmath

class OrderTracer:
    """Base class for Order tracing algorithms.
    """
    
    @staticmethod
    def gen_image(orders_list, ny, nx, sregion):
        """Generates a synthetic echellogram where the value of each pixel is the label of that trace.

        Args:
            orders_list (list): The list of dictionaries for each trace.
            shape (tuple): The dimensions of the image (ny, nx).
            xrange (list, optional): The bounding left and right pixels. Defaults to [0, nx-1].
            poly_mask_top (list, optional): A list of points (each point a list of length 2 containing an (x, y) pair) that define the bounding polynomial at the top of the image. Defaults to None.
            poly_mask_bottom (list, optional): Same but for the bottom bounding polynomial. Defaults to None.

        Returns:
            np.ndarray: The image.
        """
        
        # Initiate order image
        order_image = np.full((ny, nx), np.nan)

        # Helpful arr
        xarr = np.arange(nx)
        
        for o in range(len(orders_list)):
            order_center = np.polyval(orders_list[o]['pcoeffs'], xarr)
            for x in range(nx):
                ymid = order_center[x]
                y_low = int(np.floor(ymid - orders_list[o]['height'] / 2))
                y_high = int(np.ceil(ymid + orders_list[o]['height'] / 2))
                if y_low < 0 or y_high > nx - 1:
                    continue
                order_image[y_low:y_high + 1, x] = orders_list[o]['label']

        # Mask image
        sregion.mask_image(order_image)
                
        return order_image


class PeakTracer(OrderTracer):
    """Traces orders by looking at vertical cross sections in a smoothed image.
    """
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, deg=2, order_spacing=10, order_heights=10, xleft=None, xright=None, n_slices=10):
        """Construct a PeakTracer object.

        Args:
            deg (int, optional): The polynomial degree to fit the positions with. A higher order polynomial can be used in extraction to further refine these positions if desired. Defaults to 2.
            order_spacing (int, optional): The minimum spacing between the edges of orders. Defaults to 10.
            order_heights (int, optional): The minimum height (in spatial direction) of the orders. Defaults to 10.
            xleft (int, optional): The left most slice. Defaults to None.
            xright (int, optional): The right most slice. Defaults to None.
            n_slices (int, optional): How many slices to use. Defaults to 10.
        """
        self.deg = deg
        self.order_spacing = order_spacing
        self.xleft = xleft
        self.xright = xright
        self.n_slices = n_slices
        self.order_heights = order_heights


    ######################
    #### TRACE ORDERS ####
    ######################
        
    def trace(self, order_map, sregion, fiber=None):
        """Trace the orders.

        Args:
            recipe (ReduceRecipe): The reduce recipe.
            data (Echellogram): The data object to trace orders from.
        """

        image = order_map.parse_image()

        # dims
        ny, nx = image.shape

        # xleft and xright
        if self.xleft is None:
            self.xleft = sregion.pixmin - 1
        if self.xright is None:
            self.xright = sregion.pixmax + 1

        try:
            iter(self.order_heights)
        except:
            self.order_heights = np.full(sregion.n_orders, self.order_heights)

        # Helpful arrs
        xarr = np.arange(nx)
        yarr = np.arange(ny)
    
        # Mask
        image = np.copy(image)
        sregion.mask_image(image)

        # Smooth the image
        image_smooth = pcmath.median_filter2d(image, width=3, preserve_nans=False)

        # Slices
        xslices = np.linspace(self.xleft + 20, self.xright - 20, num=self.n_slices).astype(int)
        peaks = []
        for i in range(self.n_slices):
            x = xslices[i]
            s = np.nanmedian(image_smooth[:, x-5:x+5], axis=1)
            goody, _ = np.where(np.isfinite(image_smooth[:, x-5:x+5]))
            yi, yf = goody.min(), goody.max()
            continuum = pcmath.generalized_median_filter1d(s, width=3 * int(np.nanmean(self.order_spacing)), percentile=0.99)
            continuum[yi:yi+20] = np.nanmedian(continuum[yi+20:yi+40])
            continuum[yf-20:] = np.nanmedian(continuum[yf-40:])
            s /= continuum
            good = np.where(s > 0.7)
            s[good] = 1.0

            # Peak finding
            # Estimate peaks in pixel space (just indices)
            _peaks = scipy.signal.find_peaks(s, height=np.full(ny, 0.75), distance=self.order_spacing)[0]
            _peaks = np.sort(_peaks)
            if len(_peaks) == sregion.n_orders:
                peaks.append(_peaks)
            else:
                peaks.append(None)

        # Fit
        poly_coeffs = []
        for i in range(sregion.n_orders):
            xx = [xslices[i] for i in range(len(xslices)) if peaks[i] is not None]
            yy = [_peaks[i] for _peaks in peaks if _peaks is not None]
            pfit = np.polyfit(xx, yy, self.deg)
            poly_coeffs.append(pfit)

        # Ensure the labels are properly sorted
        y_mean = np.array([np.nanmean(np.polyval(poly_coeffs[i], xarr)) for i in range(len(poly_coeffs))], dtype=float)
        ss = np.argsort(y_mean)
        poly_coeffs = [poly_coeffs[ss[i]] for i in range(len(ss))]

        # Now build the orders list
        orders_list = []
        for i in range(len(poly_coeffs)):
            if sregion.orderbottom < sregion.ordertop:
                order = sregion.orderbottom + i
            else:
                order = sregion.orderbottom - i
            if fiber is not None:
                label = float(str(int(order)) + "." + str(int(fiber)))
            else:
                label = order
            orders_list.append({'label': label, 'height': self.order_heights[i], 'pcoeffs': poly_coeffs[i]})

        # Return
        order_map.orders_list = orders_list