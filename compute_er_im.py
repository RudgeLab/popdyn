import os
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from skimage.transform import warp_polar
from skimage.transform import warp, EuclideanTransform
from scipy.io import savemat

path = '/mnt/ff9e5a34-3696-46e4-8fa8-0171539135be/Tweez scope/2023_11_28'
fname = '2023_11_28_10x_1.0x_pAAA_TiTweez_Pos1.ome.tif'
folder_res = 'results'
folder_fluo = 'fluo'

path_im = os.path.join(path, fname)
path_res = os.path.join(path, folder_res)
path_fluo = os.path.join(path, folder_fluo)

im = imread(path_im)
im = im.astype(float)
#im = im.transpose([0,2,3,1])
nt,nx,ny,nc = im.shape
print(im.shape)

###### OLD
# Compute mean fluo at each time point
for c in range(3):
    cim = im[:,:,:,c] 
    cim[mask_all==0] = np.nan
    im[:,:,:,c] = cim
f = np.nanmean(im, axis=(1,2))
#np.save(os.path.join(path, folder_results, folder_pos, 'mean_fluo.npy'),  f)
##########

edt = np.load(os.path.join(path_res, 'edt.npy'))

area = np.load(os.path.join(path_res, 'area.npy'))
radius = np.sqrt(area/np.pi)
dsarea = savgol_filter(area, 21, 3, deriv=1)
dsradius = savgol_filter(radius, 21, 3, deriv=1)
sradius = savgol_filter(radius, 21, 3)
savemat(os.path.join(path_res, 'sradius.mat'), {'sradius':sradius})
savemat(os.path.join(path_res, 'dsradius.mat'), {'dsradius':dsradius})

idx = np.where(dsradius<0.25)[0]
tmin = 0 #idx.min()
tmax = nt #idx.min() + 200
tmin,tmax

edt = edt[tmin:tmax,:,:]
im = im[tmin:tmax,:,:,:]
nt,nx,ny,nc = im.shape

y,x = np.meshgrid(np.arange(1024), np.arange(1024))

# Register time series
edt0 = edt[-1,:,:]
cx0 = x[edt0>0].mean()
cy0 = y[edt0>0].mean()
for t in range(nt):
    print(f'Registering frame {t+1}/{nt}')
    tedt = edt[t,:,:]
    cx = y[tedt>0].mean()
    cy = x[tedt>0].mean()
    shift = [cx0 - cx, cy0 - cy]
    tform = EuclideanTransform(translation=shift)
    regtedt = warp(tedt, tform.inverse, preserve_range=True)
    edt[t,:,:] = regtedt
    for c in range(nc):
        ctim = im[t,:,:,c]
        regctim = warp(ctim, tform.inverse, preserve_range=True)
        im[t,:,:,c] = regctim
        #plt.subplot(1, nc, c+1)
        #plt.imshow(regctim)
    #plt.savefig('regim_%04d.png'%t)
    #plt.close()

edt0 = edt[-1,:,:]
xmin = x[edt0>0].min() - 32
xmax = x[edt0>0].max() + 32
ymin = y[edt0>0].min() - 32
ymax = y[edt0>0].max() + 32

edt = edt[:,xmin:xmax,ymin:ymax]
nt,nx,ny = edt.shape
x,y = np.meshgrid(np.arange(ny), np.arange(nx))

# change phase channel index
ph = im[:,xmin:xmax,ymin:ymax,3]

# BG correction and gaussian smoothing
# change fluo channels indexing
fluo = im[:,:,:,:3]
nt,nx,ny,nc = fluo.shape
bg = fluo[0,:100:,:100,:].mean(axis=(1,2))
for t in range(nt):
    print(f'Smoothing frame {t+1}/{nt}')
    for c in range(nc):
        fluo[t,:,:,c] = gaussian_filter(fluo[t,:,:,c] - bg[c], 8)
fluo[fluo<0] = 0
fluo = fluo[:,xmin:xmax,ymin:ymax,:]
nt,nx,ny,nc = fluo.shape
print(nt,nx,ny,nc)

sfluo = savgol_filter(fluo, 31, 3, axis=0)
dsfluo = savgol_filter(fluo, 31, 3, deriv=1, axis=0)
#np.save('results/sfluo.npy', sfluo)
#np.save('results/dsfluo.npy', dsfluo)
savemat(os.path.join(path_res, 'sfluo.mat'), {'sfluo':sfluo})
savemat(os.path.join(path_res, 'dsfluo.mat'), {'dsfluo':dsfluo})

del(fluo)
del(im)

#sfluo = np.load('results/sfluo.npy')
#dsfluo = np.load('results/dsfluo.npy')
#nt,nx,ny,nc = dsfluo.shape
#print(nt,nx,ny,nc)

gamma = np.log(2) / (12 * 60 / 10)
er = dsfluo + gamma * sfluo
vmin = [0]*3 #np.nanmin(er, axis=(0,1,2))
vmax = np.nanmax(er, axis=(0,1,2))
print(vmin, vmax)
corr = np.zeros((nt,))
for t in range(nt):
    print(f'Computing correlations frame {t+1}/{nt}')
    tedt = edt[t,:,:]
    x = er[t,:,:,0]
    y = er[t,:,:,1]
    x = x[tedt>0]
    y = y[tedt>0]
    x = (x - vmin[0]) / (vmax[0] - vmin[0])
    y = (y - vmin[1]) / (vmax[1] - vmin[1])
    corr[t] = np.sum(x * y) / np.sqrt(np.sum(x*x) + np.sum(y*y))
    #np.corrcoef(x[tedt>0].ravel(), y[tedt>0].ravel())[0,1]

plt.plot(corr)
plt.legend(['RFP-YFP', 'RFP-CFP', 'YFP-CFP'])
plt.show()

for t in range(nt):
    plt.figure(figsize=(9,3))
    for c in range(nc):
        plt.subplot(1,nc,c+1)
        tcdsfluo = dsfluo[t,:,:,c]
        tcdsfluo[edt[t,:,:]==0] = np.nan
        plt.imshow(tcdsfluo)
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(path_fluo, 'dsfluo_%04d.png'%t))
    plt.close()

for t in range(nt):
    plt.figure(figsize=(9,3))
    for c in range(nc):
        plt.subplot(1,nc,c+1)
        tcsfluo = sfluo[t,:,:,c]
        tcsfluo[edt[t,:,:]==0] = np.nan
        plt.imshow(tcsfluo)
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(path_fluo, 'sfluo_%04d.png'%t))
    plt.close()

for t in range(nt):
    plt.figure(figsize=(9,3))
    plt.subplot(1, nc+2, 1)
    plt.imshow(ph[t,:,:], cmap='gray')
    nter = np.zeros((nx,ny,3))
    for c in range(nc):
        ntcer = np.zeros((nx,ny,3))
        plt.subplot(1,nc+2,c+2)
        tcer = er[t,:,:,c]
        tcer[edt[t,:,:]==0] = np.nan
        #nter[:,:,c] = (tcer - np.nanmin(tcer)) / (np.nanmax(tcer)  - np.nanmin(tcer))
        nter[:,:,c] = (tcer - vmin[c]) / (vmax[c]  - vmin[c])
        ntcer[:,:,c] = nter[:,:,c]
        plt.imshow(ntcer)
        #plt.colorbar()
    plt.subplot(1, nc+2, nc+2)
    plt.imshow(nter)
    plt.tight_layout()
    plt.savefig(os.path.join(path_fluo, 'er_%04d.png'%t))
    plt.close()

'''
gamma = np.log(2) / (12 * 60 / 10)
nr,ntheta = 100,100
for t in range(nt):
    plt.figure(figsize=(9,3))
    npter = np.zeros((nr,ntheta,3))
    for c in range(nc):
        nptcer = np.zeros((nr,ntheta,3))        
        plt.subplot(1,nc+1,c+1)
        tcer = er[t,:,:,c]
        cx = x[edt[t,:,:]>0].mean()
        cy = y[edt[t,:,:]>0].mean()
        ptcer = warp_polar(tcer, center=(cx,cy), radius=edt.max(), output_shape=(nr,ntheta))
        npter[:,:,c] = (ptcer - np.nanmin(ptcer)) / (np.nanmax(ptcer)  - np.nanmin(ptcer))
        nptcer[:,:,c] = npter[:,:,c]
        plt.imshow(nptcer)
        #plt.colorbar()
    plt.subplot(1, nc+1, nc+1)
    plt.imshow(npter)
    plt.tight_layout()
    plt.savefig('per_%04d.png'%t)
    plt.close()
'''



