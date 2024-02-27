import os
import numpy as np
from skimage.io import imread, imsave
from skimage.measure import find_contours
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.morphology import binary_erosion
from skimage.draw import polygon
import matplotlib.pyplot as plt

start_frame = 0
step = 1

#path = '/home/guillermo/Microscopy/Ti scope'
path = '/mnt/ff9e5a34-3696-46e4-8fa8-0171539135be/Tweez scope/2023_11_28'
folder_masks = 'contour_masks'
fname = '2023_11_28_10x_1.0x_pAAA_TiTweez_Pos1.ome.tif'
fname_mask = '2023_11_28_10x_1.0x_pAAA_TiTweez_Pos1.ome.contour.mask.tif'

path_im = os.path.join(path, fname)
path_masks = os.path.join(path, folder_masks, fname_mask)

im_all = imread(path_im)
print(im_all.shape)
im_all = im_all[:,:,:,3]
im_all = im_all.astype(float)

nt,nx,ny = im_all.shape
# input manually from image inspection
cx,cy = 520,520
radius = 100

mask_out = np.zeros((nt,) + im_all.shape[1:3])
for t in range(nt):
    print(f'Processing frame {t+1}/{nt}')
    
    # normalize pixel values to [0, 1]
    f = im_all[start_frame + t*step,:,:]
    f = (f - f.min()) / (f.max() - f.min())
    
    # set initial contour as a circle around initial guess (cx, cy, radius)
    ang = np.linspace(0, 2*np.pi, 100)
    x = radius * np.cos(ang) + cx
    y = radius * np.sin(ang) + cy
    init = np.zeros((len(ang),2))
    init[:,0] = x
    init[:,1] = y

    # Gaussian blurring to smooth the image
    # adjust the initial contour to fit the actual boundary of the colony
    # generates the vertex coordinates of the colony
    snake = active_contour(gaussian(f, 3, preserve_range=False),
                       init, alpha=5e-3, beta=1e-6, gamma=0.001, w_edge=radius/50, w_line=0)

    mnew = np.zeros_like(f)
    # generate the coordinates of the pixels inside the polygon defined by the vertex coordinates
    rr, cc = polygon(snake[:, 0], snake[:, 1], mnew.shape)
    mnew[rr,cc] = 1

    # remove pixels on the boundaries of the contour
    for _ in range(8):
        mnew = binary_erosion(mnew)
    mask_out[t,:,:] = mnew

    # update center coordinates and radius of the contour based on the new mask
    cx,cy = snake.mean(axis=0)
    area = np.sum(mnew)
    # empirical adjustment of 50 so the contour slightly exceeds the actual boundary of the colony
    radius = np.sqrt(area/np.pi) + 50

    plt.imshow(f, cmap='gray')
    plt.plot(init[:,1], init[:,0], 'r--')
    plt.plot(snake[:,1], snake[:,0], 'g-')
    plt.savefig(os.path.join(path, folder_masks,'contour_%03d.png'%t))
    plt.close()

imsave(path_masks, mask_out>0)