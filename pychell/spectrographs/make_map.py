import numpy as np
from astropy.io import fits
import pickle
import matplotlib.pyplot as plt
from pdb import set_trace as stop
with open('chiron_order_locations_coeffs_new.pkl', 'rb') as f:
    pcoeffs = pickle.load(f)


#ny, nx = fits.open('chi190915.1242.fits')[0].data.shape

h = 10
nx, ny = 4112, 1432
xarr = np.arange(nx)

order_image = np.zeros(shape=(ny, nx), dtype=float) + np.nan
order_dicts = []
for o in range(len(pcoeffs)):
    order_dicts.append({'label': o + 1, 'pcoeffs': pcoeffs[o], 'height': h})
    ypositions = np.polyval(pcoeffs[o], xarr)
    ytop = np.round(ypositions + h / 2).astype(int)
    ybottom = np.round(ypositions - h / 2).astype(int)
    for x in range(nx):
        order_image[ybottom[x]:ytop[x], x] = o + 1
    
plt.imshow(order_image)
plt.show()
fits.writeto('chiron_order_map_master.fits', order_image, overwrite=True)
with open('chiron_order_map_dicts_master.pkl', 'wb') as f:
    pcoeffs = pickle.dump(order_dicts, f)