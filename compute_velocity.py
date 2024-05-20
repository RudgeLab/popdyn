import infotracking
from infotracking import Ensemble, infotheory
import numpy as np
import skimage
from skimage.io import imread,imsave
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# Parameters -----------------------------------
folder = '/media/c1046372/Expansion/Conor MSc/Analysis_2/DHL708/14-06-23_pLPT20/Position 0'
path = folder+'/'+'velocity_data'
startframe = 30
step = 1
nframes = 5
nt = nframes-1

windowsize = 64
windowspacing = 32
window_px0 = 0
window_py0 = 0

maxvel = 19

#------------------------------------------------
im = imread(folder+'/'+'10x_1.0x_pLPT20_DHL_1_MMStack_Pos0.ome.tif')
im = im[:,:,:,0]
mask = imread(folder+'/'+'contour_masks/120_frames/10x_1.0x_pLPT20_DHL_1_MMStack_Pos0.contour.mask.tif')
init_vel = np.zeros(mask.shape + (2,))
mask = mask / mask.max() # Make sure 0-1
im = im[startframe:startframe+(nframes * step):step,:,:]
mask = mask[startframe:startframe+(nframes * step):step,:,:]

print("Image dimensions ",im.shape)
eg = Ensemble.EnsembleGrid(im, mask, init_vel, mask_threshold=0.5)

eg.initialise_ensembles(windowsize,windowsize, \
                        windowspacing,windowspacing, \
                        window_px0,window_py0)
print("Grid dimensions ", eg.gx,eg.gy)

eg.compute_motion(nt,maxvel,maxvel,velstd=21,dt=1)


# Generate some output
print("Saving quiver plots...")
eg.save_quivers(path, 'quiver_image_%04d.png', 'quiver_plain_%04d.png', normed=False)
print("Saving data files...")
eg.save_data(path)

