# Maths
import numpy as np
import scipy.signal

# pychell
import pychell.maths as pcmath

class OrderTracer:
    """Base class for Order tracing algorithms.
    """
    
    @staticmethod
    def gen_image(orders_list, ny, nx, xrange=None, poly_mask_top=None, poly_mask_bottom=None):
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

        # Xrange
        if xrange is None:
            xrange = [0, nx - 1]
        
        # Top polynomial
        x_bottom = np.array([p[0] for p in poly_mask_bottom], dtype=float)
        y_bottom = np.array([p[1] for p in poly_mask_bottom], dtype=float)
        pfit_bottom = np.polyfit(x_bottom, y_bottom, len(x_bottom) - 1)
        poly_bottom = np.polyval(pfit_bottom, xarr)

        # Bottom polynomial
        x_top = np.array([p[0] for p in poly_mask_top], dtype=float)
        y_top = np.array([p[1] for p in poly_mask_top], dtype=float)
        pfit_top = np.polyfit(x_top, y_top, len(x_top) - 1)
        poly_top = np.polyval(pfit_top, xarr)
        
        for o in range(len(orders_list)):
            order_center = np.polyval(orders_list[o]['pcoeffs'], xarr)
            for x in range(xrange[0], xrange[1] + 1):
                ymid = order_center[x]
                y_low = int(np.floor(ymid - orders_list[o]['height'] / 2))
                y_high = int(np.ceil(ymid + orders_list[o]['height'] / 2))
                if y_low < poly_bottom[x] or y_high > poly_top[x]:
                    continue
                order_image[y_low:y_high + 1, x] = orders_list[o]['label']

        return order_image


class PeakTracer(OrderTracer):
    """Traces orders by looking at vertical cross sections in a smoothed image.
    """
    
    ###############################
    #### CONSTRUCTOR + HELPERS ####
    ###############################

    def __init__(self, orders=None, deg=2, order_spacing=10, order_heights=10, xleft=None, xright=None, n_slices=10):
        """Construct a PeakTracer object.

        Args:
            n_orders (int): The number of orders.
            deg (int, optional): The polynomial degree to fit the positions with. A higher order polynomial can be used in extraction to further refine these positions if desired. Defaults to 2.
            order_spacing (int, optional): The minimum spacing between the edges of orders. Defaults to 10.
            order_heights (int, optional): The minimum height (in spatial direction) of the orders. Defaults to 10.
            xleft (int, optional): The left most slice. Defaults to None.
            xright (int, optional): The right most slice. Defaults to None.
            n_slices (int, optional): How many slices to use. Defaults to 10.
        """
        self.orders = orders
        self.deg = deg
        self.order_spacing = order_spacing
        self.xleft = xleft
        self.xright = xright
        self.n_slices = n_slices
        try:
            iter(order_heights)
            self.order_heights = order_heights
        except:
            self.order_heights = np.full(self.orders[1] - self.orders[0] + 1, order_heights)


    ######################
    #### TRACE ORDERS ####
    ######################
        
    def trace(self, recipe, data):
        """Trace the orders.

        Args:
            recipe (ReduceRecipe): The reduce recipe.
            data (Echellogram): The data object to trace orders from.
        """

        # Image
        image = data.parse_image()

        # dims
        ny, nx = image.shape

        # xleft and xright
        if self.xleft is None:
            xleft = int(nx / 4)
        else:
            xleft = self.xleft
        if self.xright is None:
            xright = int(3 * nx / 4)
        else:
            xright = self.xright
        
        # Fiber number
        try:
            fiber = int(data.spec_module.parse_fiber_nums(data))
        except:
            fiber = None

        # Call function
        orders_list = self._trace(image, self.orders, self.deg, self.order_spacing, self.order_heights, recipe.xrange, recipe.poly_mask_bottom, recipe.poly_mask_top, fiber, xleft, xright, self.n_slices)

        # Store result
        data.orders_list = orders_list


    @staticmethod
    def _trace(image, orders, deg=2, order_spacing=None, order_heights=None, xrange=None, poly_mask_bottom=None, poly_mask_top=None, fiber=None, xleft=None, xright=None, n_slices=10):

        n_orders = orders[1] - orders[0] + 1

        try:
            iter(order_heights)
        except:
            order_heights = np.full(n_orders, order_heights)

        # Image dimensions
        ny, nx = image.shape

        # Helpful arrs
        xarr = np.arange(nx)
        yarr = np.arange(ny)

        # xrange
        if xrange is None:
            xrange = [int(0.1 * nx), int(0.9 * nx)]
    
        # Mask
        image = np.copy(image)
        image[:, 0:xrange[0]] = np.nan
        image[:, xrange[1] + 1:] = np.nan

        # Top polynomial
        x_bottom = np.array([p[0] for p in poly_mask_bottom], dtype=float)
        y_bottom = np.array([p[1] for p in poly_mask_bottom], dtype=float)
        pfit_bottom = np.polyfit(x_bottom, y_bottom, len(x_bottom) - 1)
        poly_bottom = np.polyval(pfit_bottom, xarr)

        # Bottom polynomial
        x_top = np.array([p[0] for p in poly_mask_top], dtype=float)
        y_top = np.array([p[1] for p in poly_mask_top], dtype=float)
        pfit_top = np.polyfit(x_top, y_top, len(x_top) - 1)
        poly_top = np.polyval(pfit_top, xarr)

        for x in range(nx):
            bad = np.where((yarr < poly_bottom[x]) | (yarr > poly_top[x]))[0]
            image[bad, x] = np.nan

        # Smooth the flat
        image_smooth = pcmath.median_filter2d(image, width=3, preserve_nans=False)

        # Slices
        xslices = np.linspace(xleft + 20, xright - 20, num=n_slices).astype(int)
        peaks = []
        for i in range(n_slices):
            x = xslices[i]
            s = np.nanmedian(image_smooth[:, x-5:x+5], axis=1)
            goody, _ = np.where(np.isfinite(image_smooth[:, x-5:x+5]))
            yi, yf = goody.min(), goody.max()
            continuum = pcmath.generalized_median_filter1d(s, width=3 * int(np.nanmean(order_spacing)), percentile=0.99)
            continuum[yi:yi+20] = np.nanmedian(continuum[yi+20:yi+40])
            continuum[yf-20:] = np.nanmedian(continuum[yf-40:])
            s /= continuum
            good = np.where(s > 0.7)
            s[good] = 1.0

            # Peak finding
            # Estimate peaks in pixel space (just indices)
            _peaks = scipy.signal.find_peaks(s, height=np.full(ny, 0.75), distance=order_spacing)[0]
            _peaks = np.sort(_peaks)
            if len(_peaks) == n_orders:
                peaks.append(_peaks)
            else:
                peaks.append(None)

        # Fit
        poly_coeffs = []
        for i in range(n_orders):
            xx = [xslices[i] for i in range(len(xslices)) if peaks[i] is not None]
            yy = [_peaks[i] for _peaks in peaks if _peaks is not None]
            pfit = np.polyfit(xx, yy, deg)
            poly_coeffs.append(pfit)

        # Ensure the labels are properly sorted
        y_mean = np.array([np.nanmean(np.polyval(poly_coeffs[i], xarr)) for i in range(len(poly_coeffs))], dtype=float)
        ss = np.argsort(y_mean)
        poly_coeffs = [poly_coeffs[ss[i]] for i in range(len(ss))]

        # Now build the orders list
        orders_list = []
        for i in range(len(poly_coeffs)):
            if fiber is not None:
                label = float(str(int(i + orders[0])) + "." + str(int(fiber)))
            else:
                label = i + orders[0]
            orders_list.append({'label': label, 'height': order_heights[i], 'pcoeffs': poly_coeffs[i]})
        
        return orders_list
