import numpy as np
import os
from skimage.io import imread, imsave
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

# Image frame to start from and step between frames
start_frame = 0
step = 1
nframes = 70

position = 0
path = '/media/c1046372/Expansion/Thesis GY/3. Analyzed files'
scope_name = 'Ti scope'
exp_date = '2023_11_15'
velocity_folder = 'velocity_data'
masks_folder = 'contour_masks'
results_folder = 'results'
dnas = {'pLPT20&pLPT41': 'pLPT20&41', 'pLPT119&pLPT41': 'pLPT119&41', 'pAAA': 'pAAA', 'pLPT107&pLPT41': 'pLPT107&41'}
scopes = {'Tweez scope': 'TiTweez', 'Ti scope': 'Ti'}
vector = 'pLPT20&pLPT41'




vel = np.load(os.path.join(path,scope_name,exp_date,velocity_folder,f'pos{position}','vel.np.npy'))
pos = np.load(os.path.join(path,scope_name,exp_date,velocity_folder,f'pos{position}','pos.np.npy'))

# Size of data
nx, ny, nt, _ = vel.shape

fname = f'{exp_date}_10x_1.0x_{dnas[vector]}_{scopes[scope_name]}_Pos{position}.ome.tif'
#im_all = imread(f'/media/c1046372/Expansion/Thesis GY/3. Analyzed files/Ti scope/2023_11_15/2023_11_15_10x_1.0x_{}_Ti_Pos0.ome.tif')
edt = np.load(os.path.join(path,scope_name,exp_date,results_folder,f'pos{position}','edt.npy'))
mask_all = imread(os.path.join(path,scope_name,exp_date,masks_folder,'mask_'+fname))
mask_all = mask_all > 0

_, edtnx, edtny = edt.shape
# mask_all = np.zeros(im_all.shape[:3])
x, y = np.meshgrid(np.arange(edtnx), np.arange(edtny))

# Make arrays to store results
radpos = np.zeros((nt, nx, ny))
vmag = np.zeros((nt, nx, ny))
vrad = np.zeros((nt, nx, ny))
vtheta = np.zeros((nt, nx, ny))

# Process the data and save results
for frame in range(nt):
    print(f'Processing frame {frame}')

    mask = mask_all[start_frame + frame * step + 1, :, :]
    cx = x[mask > 0].mean()
    cy = y[mask > 0].mean()

    vx = vel[:, :, frame, 0]
    vy = vel[:, :, frame, 1]

    # Subtract drift from velocities
    vx -= np.nanmean(vx)
    vy -= np.nanmean(vy)

    # Get direction to colony edge as negative of gradient of distance
    gradx, grady = np.gradient(edt[start_frame + frame * step + 1, :, :])
    gradx[mask == 0] = np.nan
    grady[mask == 0] = np.nan
    px = pos[:, :, frame, 0].astype(int)
    py = pos[:, :, frame, 1].astype(int)
    pnorm = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)

    gx = np.zeros((nx, ny))
    gy = np.zeros((nx, ny))
    for ix in range(nx):
        for iy in range(ny):
            gx[ix, iy] = -np.nanmean(gradx[px[ix, iy]:px[ix, iy] + 64, py[ix, iy]:py[ix, iy] + 64])
            gy[ix, iy] = -np.nanmean(grady[px[ix, iy]:px[ix, iy] + 64, py[ix, iy]:py[ix, iy] + 64])
            # radpos[frame,ix,iy] = np.nanmean(edt[frame, px[ix,iy]-32:px[ix,iy]+32, py[ix,iy]-32:py[ix,iy]+32])
    # Compute magnitude of velocities in radial direction
    velnorm = np.sqrt(vx ** 2 + vy ** 2)
    gnorm = np.sqrt(gx ** 2 + gy ** 2)
    vmag[frame, :, :] = vx * gx + vy * gy
    vrad[frame, :, :] = vmag[frame, :, :] / velnorm / gnorm
    vperp = vx * gy - vy * gx
    vtheta[frame, :, :] = vperp / velnorm / gnorm

    # Radial position of each grid square
    radpos[frame, :, :] = edt[frame, px + 16, py + 16]

# Save results

path_save = os.path.join(path,scope_name,exp_date,results_folder,f'pos{position}')
np.save(os.path.join(path_save,'radpos.npy'), radpos)
np.save(os.path.join(path_save,'vmag.npy'), vmag)
np.save(os.path.join(path_save,'vrad.npy'), vrad)
np.save(os.path.join(path_save,'vtheta.npy'), vtheta)
