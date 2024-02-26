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

fname = '10x_1.0x_pAAA_TiTweez_1_MMStack_Pos0_mer.ome.tif'
im_all = imread(fname)
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
    #f = im_all[start_frame + t*step,:,:,1]
    f = im_all[start_frame + t*step,:,:]
    f = (f - f.min()) / (f.max() - f.min())
    #m = mask[start_frame + t*step,:,:]
    #init = find_contours(m, 0.5)[0]
    
    ang = np.linspace(0, 2*np.pi, 100)
    w,h = f.shape[:2]
    #radius = 2 * t * (min(w/2, h/2) - 150) / nt + 150
    #radius = min(min(w/2, h/2), radius)
    x = radius * np.cos(ang) + cx
    y = radius * np.sin(ang) + cy
    init = np.zeros((len(ang),2))
    init[:,0] = x
    init[:,1] = y

    #snake = active_contour(gaussian(f, 3, preserve_range=False),
    #                   init, alpha=0.03, beta=20, gamma=0.001, w_edge=5, w_line=0)
    snake = active_contour(gaussian(f, 3, preserve_range=False),
                       init, alpha=5e-3, beta=1e-6, gamma=0.001, w_edge=radius/50, w_line=0)

    mnew = np.zeros_like(f)
    rr, cc = polygon(snake[:, 0], snake[:, 1], mnew.shape)
    mnew[rr,cc] = 1

    for _ in range(8):
        mnew = binary_erosion(mnew)
    mask_out[t,:,:] = mnew

    cx,cy = snake.mean(axis=0)
    area = np.sum(mnew)
    radius = np.sqrt(area/np.pi) + 50

    plt.imshow(f, cmap='gray')
    plt.plot(init[:,1], init[:,0], 'r--')
    plt.plot(snake[:,1], snake[:,0], 'g-')
    plt.savefig('contour_masks/contour_%03d.png'%t)
    plt.close()

imsave('./contour_masks/10x_1.0x_pAAA_TiTweez_1_MMStack_Pos0_mer.ome.contour.mask.tif', mask_out>0)
