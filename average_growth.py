import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.optimize import fmin, least_squares
from scipy.signal import savgol_filter

step = 1

im = imread('10x_1.0x_pAAA_TiTweez_1_MMStack_Pos0_mer.ome.tif')
# fluo channels
im = im[:,:,:,:3].astype(float)
# BG correction using first frame, too much scattering in later frames
bg = im[0,:100:,:100,:].mean(axis=(1,2))
nt,nx,ny,nc = im.shape
for t in range(nt):
    for c in range(nc):
        im[t,:,:,c] = im[t,:,:,c] - bg[t,c]
im[im<0] = 0

mask_all = imread('contour_masks/10x_1.0x_pAAA_TiTweez_1_MMStack_Pos0_mer.ome.contour.mask.tif')
mask_all = mask_all>0
nt,nx,ny = mask_all.shape
area = mask_all[:nt*step:step,:,:].sum(axis=(1,2))
radius = np.sqrt(area / np.pi)

rmax = np.zeros_like(radius)
for t in range(nt):
    m = mask_all[t*step,:,:]
    edt = distance_transform_edt(m)
    rmax[t] = edt.max()

#p = np.polyfit(np.arange(40), radius[10:50], 1)
#vfront = p[0]
#print(f'vfront = {vfront}')
vfront = savgol_filter(radius, 11, 3, deriv=1)
np.save('results/vfront.npy', vfront)
exprate = savgol_filter(area, 11, 3, deriv=1) / savgol_filter(area, 11, 3)
np.save('results/area_gr.npy', exprate)

plt.subplot(4,1,1)
plt.plot(area)
np.save('results/area.npy', area)
plt.subplot(4,1,2)
plt.plot(radius)
np.save('results/radius.npy', radius)
plt.subplot(4,1,3)
plt.plot(vfront)
plt.subplot(4,1,4)
plt.plot(exprate)
plt.savefig('graphs/a_r_vf_exp.png')
plt.show()

edt = np.zeros_like(mask_all).astype(float)
for t in range(nt):
    edt[t,:,:] = distance_transform_edt(mask_all[t,:,:])
np.save('results/edt.npy',  edt)

# Compute mean fluo at each time point
for c in range(3):
    cim = im[:,:,:,c] 
    cim[mask_all==0] = np.nan
    im[:,:,:,c] = cim
f = np.nanmean(im, axis=(1,2))
print(f)
plt.figure()
plt.plot(f)
plt.savefig('graphs/mean_fluo.png')
plt.show()

plt.figure()
plt.plot(f[:,1], f[:,0])
plt.savefig('graphs/phase_plot.png')
plt.show()

plt.figure()
plt.plot(f[:,0] / f[:,1])
plt.savefig('graphs/ratio.png')
plt.show()
