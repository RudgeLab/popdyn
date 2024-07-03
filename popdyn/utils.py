import os
import json
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2
matplotlib.use('Agg')

from skimage.io import imread, imsave
from skimage.measure import find_contours
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.morphology import binary_erosion
from skimage.draw import polygon
from skimage.transform import warp_polar
from skimage.transform import warp, EuclideanTransform

from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import fmin, least_squares
from scipy.signal import savgol_filter
from scipy.io import savemat
import infotracking
from infotracking import Ensemble, infotheory

def make_video(images_folder,  video_name):
    """
    Makes a video from a sequence of png images.
    
    Parameters:
    - images_folder (string): Folder that contains the png files.
    - word_cont (string): Word that the files contain.
    - video_name (string): Name of the output file.

    Returns:
    - 
    """
    images = [img for img in os.listdir(images_folder) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(images_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 7, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(images_folder, image)))

    video.release()

def bg_corr(im, rmin, rmax, cmin, cmax):
    # now it makes bg correction using just the first pixel because of later scattering
    bg = im[0,rmin:rmax, cmin:cmax,:].mean(axis=(1,2))
    nt,nx,ny,nc = im.shape
    for t in range(nt):
        print(f'Smoothing frame {t+1}/{nt}')
        for c in range(nc):
            im[t,:,:,c] = gaussian_filter(im[t,:,:,c] - bg[c], 8)
    im[im<0] = 0

    return im

def contour_mask(im_ph, start_frame, step, pos, cx, cy, radius, path, folder_masks, path_masks, radj):
    nt,nx,ny = im_ph.shape
    mask_out = np.zeros((nt,) + (nx,ny))

    temp_folder = os.path.join(path, folder_masks,f"temp_pos{pos}")
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    for t in range(nt):
        print(f'Processing frame {t+1}/{nt}')
        
        # normalize pixel values to [0, 1]
        f = im_ph[start_frame + t*step,:,:]
        f = (f - f.min()) / (f.max() - f.min())
        # set initial contour as a circle around initial guess (cx, cy, radius)
        ang = np.linspace(0, 2*np.pi, 100)
        x = radius * np.cos(ang) + cx
        y = radius * np.sin(ang) + cy
        init = np.zeros((len(ang),2))
        init[:,0] = x
        init[:,1] = y

        # Gaussian blur to smooth the image
        # adjust the initial contour to fit the actual boundary of the colony
        # generates the vertex coordinates of the colony
        snake = active_contour(gaussian(f, 3, preserve_range=False),
                        # for Tweezers 12_06
                        init, alpha=5e-3, beta=1e-6, gamma=0.001, w_edge=radius/30, w_line=0)
    #                    init, alpha=5e-3, beta=1e-6, gamma=0.001, w_edge=radius/50, w_line=0)

        mnew = np.zeros_like(f)
        # generate the coordinates of the pixels inside the polygon defined by the vertex coordinates
        rr, cc = polygon(snake[:, 0], snake[:, 1], mnew.shape)
        mnew[rr,cc] = 1

        """
        # remove pixels on the boundaries of the contour
        for _ in range(8):
            mnew = binary_erosion(mnew)
        """
        mask_out[t,:,:] = mnew
        
        # update center coordinates and radius of the contour based on the new mask
        cx,cy = snake.mean(axis=0)
        area = np.sum(mnew)
        # empirical adjustment of 50 so the contour slightly exceeds the actual boundary of the colony
        #radius = np.sqrt(area/np.pi) + 50
        radius = np.sqrt(area/np.pi) + radj

        plt.imshow(f, cmap='gray')
        plt.plot(init[:,1], init[:,0], 'r--')
        plt.plot(snake[:,1], snake[:,0], 'g-')
        plt.savefig(os.path.join(path, folder_masks,f"temp_pos{pos}",'contour_%03d.png'%t))
        plt.close()

    imsave(path_masks, mask_out>0)
    make_video(os.path.join(path, folder_masks,f"temp_pos{pos}"), 
           os.path.join(path, folder_masks,f"pos{pos}_contour.avi"))

def average_growth(path_masks, step, pos, path, folder_results, folder_graphs):
    
    folder_pos = os.path.join(path, folder_results,f"pos{pos}")
    if not os.path.exists(folder_pos):
        os.makedirs(folder_pos)

    folder_pos_graph = os.path.join(path, folder_graphs,f"pos{pos}")
    if not os.path.exists(folder_pos_graph):
        os.makedirs(folder_pos_graph)

    mask_all = imread(path_masks)
    mask_all = mask_all>0
    nt,nx,ny = mask_all.shape
    area = mask_all[:nt*step:step,:,:].sum(axis=(1,2))
    radius = np.sqrt(area / np.pi)

    
    vfront = savgol_filter(radius, 11, 3, deriv=1)
    exprate = savgol_filter(area, 11, 3, deriv=1) / savgol_filter(area, 11, 3)
    edt = np.zeros_like(mask_all).astype(float)
    for t in range(nt):
        edt[t,:,:] = distance_transform_edt(mask_all[t,:,:])

    np.save(os.path.join(folder_pos, 'radius.npy'), radius)
    np.save(os.path.join(folder_pos, 'area.npy'), area) 
    np.save(os.path.join(folder_pos, 'vfront.npy'), vfront)
    np.save(os.path.join(folder_pos, 'expansion_rate.npy'), exprate)
    np.save(os.path.join(folder_pos, 'edt.npy'),  edt)

    plt.subplot(4,1,1)
    plt.plot(area)
    plt.subplot(4,1,2)
    plt.plot(radius)
    plt.subplot(4,1,3)
    plt.plot(vfront)
    plt.subplot(4,1,4)
    plt.plot(exprate)
    plt.savefig(os.path.join(folder_pos_graph, 'a_r_vf_exp.png'))
    plt.close()

def register(im, edt):
    nt,nx,ny,nc = im.shape
    y,x = np.meshgrid(np.arange(ny), np.arange(nx))

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
        
        # register euclidean distance
        regtedt = warp(tedt, tform.inverse, preserve_range=True)
        edt[t,:,:] = regtedt
        
        # register all channels
        for c in range(nc):
            ctim = im[t,:,:,c]
            regctim = warp(ctim, tform.inverse, preserve_range=True)
            im[t,:,:,c] = regctim
            #plt.subplot(1, nc, c+1)
            #plt.imshow(regctim)
        #plt.savefig('regim_%04d.png'%t)
        #plt.close()
    return im, edt

def compute_er(im, pos, path, folder_results, fname, ph_chn):
    folder_pos = os.path.join(path, folder_results, f"pos{pos}")
    nt, nx, ny, nc = im.shape
    # TO DO: receive tmin and tmax as args
    tmin = 0
    tmax = nt

    area = np.load(os.path.join(folder_pos, 'area.npy'))
    radius = np.sqrt(area/np.pi)

    dsarea = savgol_filter(area, 21, 3, deriv=1)
    sradius = savgol_filter(radius, 21, 3)
    dsradius = savgol_filter(radius, 21, 3, deriv=1)

    np.save(os.path.join(folder_pos, 'dsarea.npy'), dsarea)
    np.save(os.path.join(folder_pos, 'sradius.npy'), sradius)
    np.save(os.path.join(folder_pos, 'dsradius.npy'), dsradius)
    #savemat(os.path.join(path_res, 'sradius.mat'), {'sradius':sradius})
    #savemat(os.path.join(path_res, 'dsradius.mat'), {'dsradius':dsradius})

    edt = np.load(os.path.join(folder_pos, 'edt.npy'))
    edt = edt[tmin:tmax,:,:]
    im = im[tmin:tmax,:,:,:]
    nt,nx,ny,nc = im.shape

    # register image
    im, edt = register(im, edt)
    edt_reg = edt
    # save im and edt registered
    np.save(os.path.join(folder_pos, 'im_reg.npy'), im)
    #imsave(os.path.join(folder_pos, 'registered_' + fname), im)
    np.save(os.path.join(folder_pos, 'edt_reg.npy'), edt_reg)
    
    # test
    #imsave(os.path.join(path, folder_results, folder_pos, 'im_reg.ome.tif'), im)

    # select a ROI of the image to analyze
    y,x = np.meshgrid(np.arange(nx), np.arange(ny))
    edt0 = edt[-1,:,:]
    xmin = x[edt0>0].min() - 32
    xmax = x[edt0>0].max() + 32
    ymin = y[edt0>0].min() - 32
    ymax = y[edt0>0].max() + 32

    edt = edt[:,xmin:xmax,ymin:ymax]
    nt,nx,ny = edt.shape
    x,y = np.meshgrid(np.arange(ny), np.arange(nx))

    # change phase channel index
    ph = im[:,xmin:xmax,ymin:ymax,ph_chn]

    # BG correction and gaussian smoothing
    fluo = im[:,:,:,:ph_chn]
    fluo = bg_corr(fluo, 0, 100, 0, 100)

    print("BG done")

    fluo = fluo[:,xmin:xmax,ymin:ymax,:]
    nt,nx,ny,nc = fluo.shape

    sfluo = savgol_filter(fluo, 31, 3, axis=0)
    dsfluo = savgol_filter(fluo, 31, 3, deriv=1, axis=0)
    print("savgol done to fluo")
    np.save(os.path.join(folder_pos, 'sfluo.npy'), sfluo)
    np.save(os.path.join(folder_pos, 'dsfluo.npy'), dsfluo)
    print("fluo files saved")
    #savemat(os.path.join(path, folder_results, folder_pos, 'sfluo.mat'), {'sfluo':sfluo})
    #savemat(os.path.join(path, folder_results, folder_pos, 'dsfluo.mat'), {'dsfluo':dsfluo})

    #############################################################################
    ### Expression rate computation
    gamma = np.log(2) / (12 * 60 / 10)
    er = dsfluo + gamma * sfluo
    vmin = [0]*3 #np.nanmin(er, axis=(0,1,2))
    vmax = np.nanmax(er, axis=(0,1,2))
    corr = np.zeros((nt,))
    for t in range(nt):
        print(f'Computing correlations frame {t+1}/{nt}')
        # TO DO: use edt_reg instead !!
        tedt = edt[t,:,:]
        x = er[t,:,:,0]
        y = er[t,:,:,1]
        x = x[tedt>0]
        y = y[tedt>0]
        x = (x - vmin[0]) / (vmax[0] - vmin[0])
        y = (y - vmin[1]) / (vmax[1] - vmin[1])
        corr[t] = np.sum(x * y) / np.sqrt(np.sum(x*x) + np.sum(y*y))
        #np.corrcoef(x[tedt>0].ravel(), y[tedt>0].ravel())[0,1]
    np.save(os.path.join(folder_pos, 'er.npy'), er)
    np.save(os.path.join(folder_pos, 'corr.npy'), corr)

    return er, edt_reg, sfluo, dsfluo

def plot_er(im_ph, pos, path, folder_fluo, er, edt, sfluo, dsfluo, fluo_chns):
    folder_pos = os.path.join(path, folder_fluo, f"pos{pos}")
    if not os.path.exists(folder_pos):
        os.makedirs(folder_pos)   
    
    if not os.path.exists(os.path.join(folder_pos, 'sfluo')):
        os.makedirs(os.path.join(folder_pos, 'sfluo'))
    if not os.path.exists(os.path.join(folder_pos, 'dsfluo')):
        os.makedirs(os.path.join(folder_pos, 'dsfluo'))
    if not os.path.exists(os.path.join(folder_pos, 'er')):
        os.makedirs(os.path.join(folder_pos, 'er'))

    vmin = [0]*3
    vmax = np.nanmax(er, axis=(0,1,2))

    nt, nx, ny = edt.shape
    y,x = np.meshgrid(np.arange(nx), np.arange(ny))
    edt0 = edt[-1,:,:]
    xmin = x[edt0>0].min() - 32
    xmax = x[edt0>0].max() + 32
    ymin = y[edt0>0].min() - 32
    ymax = y[edt0>0].max() + 32

    edt = edt[:,xmin:xmax,ymin:ymax]
    ph = im_ph[:,xmin:xmax,ymin:ymax]

    nt, nx, ny = edt.shape
    _, _, _, nc = sfluo.shape

    if fluo_chns == 3:
        fluo_leg = ['RFP', 'YFP', 'CFP']
        leg = ['Phase','RFP', 'YFP','CFP','All fluo']
    else:
        fluo_leg = ['YFP', 'CFP']
        leg = ['Phase','YFP','CFP','All fluo']

    for t in range(nt):
        print(f"Plotting dsfluo {t+1} / {nt}")
        plt.figure(figsize=(9,3))
        for c in range(nc):
            plt.subplot(1,nc,c+1)
            tcdsfluo = dsfluo[t,:,:,c]
            tcdsfluo[edt[t,:,:]==0] = np.nan
            plt.title(fluo_leg[c])
            plt.imshow(tcdsfluo)
            plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(folder_pos, 'dsfluo', 'dsfluo_%04d.png'%t))
        plt.close()

    for t in range(nt):
        print(f"Plotting sfluo {t+1} / {nt}")
        plt.figure(figsize=(9,3))
        for c in range(nc):
            plt.subplot(1,nc,c+1)
            tcsfluo = sfluo[t,:,:,c]
            tcsfluo[edt[t,:,:]==0] = np.nan
            plt.title(fluo_leg[c])
            plt.imshow(tcsfluo)
            plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(folder_pos, 'sfluo', 'sfluo_%04d.png'%t))
        plt.close()
        
    #leg = ['Phase','YFP','CFP','All fluo']
    for t in range(nt):
        print(f"Plotting er {t+1} / {nt}")
        plt.figure(figsize=(9,3))
        plt.subplot(1, nc+2, 1)
        plt.imshow(ph[t,:,:], cmap='gray')
        plt.title(leg[0])
        nter = np.zeros((nx,ny,3))
        for c in range(nc):
            ntcer = np.zeros((nx,ny,3))
            plt.subplot(1,nc+2,c+2)
            plt.title(leg[c+1])
            tcer = er[t,:,:,c]
            tcer[edt[t,:,:]==0] = np.nan
            #nter[:,:,c] = (tcer - np.nanmin(tcer)) / (np.nanmax(tcer)  - np.nanmin(tcer))
            nter[:,:,c] = (tcer - vmin[c]) / (vmax[c]  - vmin[c])
            ntcer[:,:,c] = nter[:,:,c]
            plt.imshow(ntcer)
            #plt.colorbar()
        plt.subplot(1, nc+2, nc+2)
        plt.title(leg[-1])
        plt.imshow(nter)
        plt.tight_layout()
        plt.savefig(os.path.join(folder_pos, 'er', 'er_%04d.png'%t))
        plt.close()

def compute_velocity(startframe, nframes, im, path_masks, path, folder_velocity, pos, windowsize, windowspacing):
    folder_pos = os.path.join(path, folder_velocity,f"pos{pos}")
    if not os.path.exists(folder_pos):
        os.makedirs(folder_pos)
    
    step = 1
    nt = nframes-1
    window_px0 = 0
    window_py0 = 0

    maxvel = 19
    mask = imread(path_masks)
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
    eg.save_quivers(folder_pos, 'quiver_image_%04d.png', 'quiver_plain_%04d.png', normed=False)
    print("Saving data files...")
    eg.save_data(folder_pos)

def process_velocity(path, fname, folder_velocity, folder_results, folder_masks, position, start_frame, step):
    vel = np.load(os.path.join(path,folder_velocity,f'pos{position}','vel.np.npy'))
    pos = np.load(os.path.join(path,folder_velocity,f'pos{position}','pos.np.npy'))

    # Size of data
    nx, ny, nt, _ = vel.shape
    edt = np.load(os.path.join(path,folder_results,f'pos{position}','edt.npy'))
    mask_all = imread(os.path.join(path,folder_masks,'mask_'+fname))
    mask_all = mask_all > 0

    _, edtnx, edtny = edt.shape
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

        path_save = os.path.join(path,folder_results,f'pos{position}')
        np.save(os.path.join(path_save,'radpos.npy'), radpos)
        np.save(os.path.join(path_save,'vmag.npy'), vmag)
        np.save(os.path.join(path_save,'vrad.npy'), vrad)
        np.save(os.path.join(path_save,'vtheta.npy'), vtheta)

def crop_image(im_all, edt, nx, ny, pad):
    y,x = np.meshgrid(np.arange(nx), np.arange(ny))
    edt0 = edt[-1,:,:]
    minx = x[edt0>0].min()
    maxx = x[edt0>0].max()
    miny = y[edt0>0].min()
    maxy = y[edt0>0].max()
    w = maxx - minx
    w = int(w//2) * 2
    h = maxy - miny
    h = int(h//2) * 2

    #print(w,h)
    #pad = 32

    #crop_im_all = np.zeros((nt,w+2*pad-1,h+2*pad-1,nc))
    #crop_edt = np.zeros((nt,w+2*pad-1,h+2*pad-1))
    crop_im_all = np.zeros((nt,w+2*pad,h+2*pad,nc))
    crop_edt = np.zeros((nt,w+2*pad,h+2*pad))
    #print(crop_im_all.shape)
    #print(crop_edt.shape)

    for t in range(nt):
        tedt = edt[t,:,:]
        cx = int(x[tedt>0].mean())
        cy = int(y[tedt>0].mean())
        cim = im_all[t,cx - w//2 - pad:cx + w//2 + pad,cy - h//2 - pad:cy + h//2 + pad,:]
        crop_im_all[t,:,:,:] = cim
        crop_edt[t,:,:] = tedt[cx - w//2 - pad:cx + w//2 + pad,cy - h//2 - pad:cy + h//2 + pad]

    return crop_im_all, crop_edt

def map_edt(p, edt, rs):
    rstep = np.mean(np.diff(rs))
    npos = p.shape[0]
    for i in range(npos):
        r,t = p[i,:].astype(int)
        # transformation from distance from the center
        p[i,0] = (edt[t,:,:].max() - rs[r]) / rstep
    return p

def compute_kymo(im_fluo, edt, nr, rw):
    # TO DO: performs only on fluo channels, so change nameto im_fluo
    nt,nx,ny,nc = im_fluo.shape
    # dont construct the kymo from the edge, so it starts from rw to edt.max
    # the edge is not that reliable because of the mask, niose numbers at the edge
    rs = np.linspace(rw, edt.max(), nr)
    kymo = np.zeros((nt,nr,nc)) + np.nan
    #nkymo = np.zeros((nt,nr,3)) + np.nan
    for t in range(nt):
        for c in range(nc):
            for ri in range(nr):
                tedt = edt[t,:,:]
                idx = np.abs(tedt - rs[ri]) < rw
                if np.sum(idx)>0:
                    ntcim = im_fluo[t,:,:,c]
                    kymo[t,ri,c] = np.nanmean(ntcim[idx])
                    #ntcnim = normed_im[t,:,:,c]
                    #nkymo[t,ri,c] = np.nanmean(ntcnim[idx])
    return kymo

def compute_dlkymo(kymo, nr, fluo_chns):    
    if fluo_chns == 3:
        kymo_rho = np.stack([kymo[:,:,0] / kymo[:,:,2], kymo[:,:,1] / kymo[:,:,2]], axis=2)
        cs = 2
    else:
        kymo_rho = kymo[:,:,0] / kymo[:,:,1]
        cs = 1

    lkymo_rho = np.log(kymo_rho)
    dlkymo_rho = np.zeros_like(lkymo_rho) + np.nan

    if fluo_chns == 3:
        for r in range(nr):
            for c in range(cs):
                idx = ~np.isnan(lkymo_rho[:,r,c])
                dlkymo_rho[idx,r,c] = savgol_filter(lkymo_rho[idx,r,c], 21, 3, deriv=1, axis=0)        
    else:
        for r in range(nr):
            idx = ~np.isnan(lkymo_rho[:,r])
            dlkymo_rho[idx,r] = savgol_filter(lkymo_rho[idx,r], 21, 3, deriv=1, axis=0)
    
    return dlkymo_rho

def plot_fluo_ratio(dlkymo_rho, edt, path, rs, pos, fluo_chns,t0):
    path_save = os.path.join(path, 'graphs', f'wdlkymo_rho_pos{pos}.png')
    wdlkymo_rho = np.zeros_like(dlkymo_rho)
    nt, _, _ = dlkymo_rho.shape
    if fluo_chns == 3:
        for c in range(fluo_chns-1):
            wdlkymo_rho[:,:,c] = warp(dlkymo_rho[:,:,c], map_edt, {'edt':edt, 'rs':rs})
        wdlkymo_rho[np.isnan(dlkymo_rho)] = np.nan
        
        plt.figure(figsize=(7,2))
        plt.subplot(1, 2, 1)
        plt.imshow(np.hstack([wdlkymo_rho[t0:,::-1,0],wdlkymo_rho[t0:,:,0]]).transpose(), 
                aspect='auto', 
                extent=[t0,nt,-edt.max(),edt.max()],
                vmin=-0.1, vmax=0.1,
                cmap='jet')
        plt.colorbar()
        plt.xlabel('Time (frames)')
        plt.ylabel('Radial distance (pixels)')
        plt.title('$d\mathrm{log}\\rho_{rc}/dt$')
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.hstack([wdlkymo_rho[t0:,::-1,1],wdlkymo_rho[t0:,:,1]]).transpose(), 
                aspect='auto', 
                extent=[t0,nt,-edt.max(),edt.max()],
                vmin=-0.1, vmax=0.1,
                cmap='jet')
        plt.colorbar()
        plt.xlabel('Time (frames)')
        plt.ylabel('Radial distance (pixels)')
        plt.title('$d\mathrm{log}\\rho_{yc}/dt$')
        
        plt.tight_layout()
        plt.savefig(path_save, dpi=300)

    else:
        wdlkymo_rho[:,:] = warp(dlkymo_rho[:,:], map_edt, {'edt':edt, 'rs':rs})
        wdlkymo_rho[np.isnan(dlkymo_rho)] = np.nan
        
        plt.figure(figsize=(7,2))
        #plt.subplot(1, 2, 1)
        plt.imshow(np.hstack([wdlkymo_rho[t0:,::-1],wdlkymo_rho[t0:,:]]).transpose(), 
                aspect='auto', 
                extent=[t0,nt,-edt.max(),edt.max()],
                vmin=-0.1, vmax=0.1,
                cmap='jet')
        plt.colorbar()
        plt.xlabel('Time (frames)')
        plt.ylabel('Radial distance (pixels)')
        plt.title('$d\mathrm{log}\\rho_{yc}/dt$')
        plt.tight_layout()
        plt.savefig(path_save, dpi=300)
    return wdlkymo_rho

def compute_corr(im_all, edt, nr, rw, fluo_chns, rfp_chn, yfp_chn, cfp_chn, path_results):    
    nt,nx,ny,nc = im_all.shape
    rs = np.linspace(rw, edt.max(), nr)
    bg = np.zeros((nc,))
    for c in range(nc):
        bg[c] = im_all[0,:100,:100,c].mean()

    cov = np.zeros((nt,nr,nc,nc))
    corr = np.zeros((nt,nr,nc))
    mean = np.zeros((nt,nr,nc))
    for t in range(nt):
        for ri in range(nr):
            tedt = edt[t,:,:]
            idx = np.abs(tedt - rs[ri]) < rw
            if np.sum(idx)>0:
                #plt.figure()
                if fluo_chns == 3:
                    ntim0 = im_all[t,:,:,rfp_chn].astype(float) - bg[rfp_chn]
                    ntim1 = im_all[t,:,:,yfp_chn].astype(float) - bg[yfp_chn]
                    ntim2 = im_all[t,:,:,cfp_chn].astype(float) - bg[cfp_chn]
                    x,y,z = ntim0[idx], ntim1[idx], ntim2[idx]
                    C = np.cov(np.stack([x, y, z]))
                    cov[t,ri,:,:] = C
                    corr[t,ri,0] = np.corrcoef(x, y)[0,1]
                    corr[t,ri,1] = np.corrcoef(x, z)[0,1]
                    corr[t,ri,2] = np.corrcoef(y, z)[0,1]
                    mean[t,ri,rfp_chn] = x.mean()
                    mean[t,ri,yfp_chn] = y.mean()
                    mean[t,ri,cfp_chn] = z.mean()
                elif fluo_chns == 2:                
                    ntim0 = im_all[t,:,:,yfp_chn].astype(float) - bg[yfp_chn]
                    ntim1 = im_all[t,:,:,cfp_chn].astype(float) - bg[cfp_chn]               
                    x,y = ntim0[idx], ntim1[idx]
                    C = np.cov(np.stack([x, y]))
                    cov[t,ri,:,:] = C
                    corr[t,ri,0] = np.corrcoef(x, y)[0,1]
                    corr[t,ri,1] = np.corrcoef(y, x)[0,1]
                    mean[t,ri,yfp_chn] = x.mean()
                    mean[t,ri,cfp_chn] = y.mean()
    
    corr_map = np.zeros_like(corr)
    mean_map = np.zeros_like(mean)
    for c in range(fluo_chns):
        corr_map[:,:,c] = warp(corr[:,:,c], map_edt, {'edt':edt, 'rs':rs})
        mean_map[:,:,c] = warp(mean[:,:,c], map_edt, {'edt':edt, 'rs':rs})
    corr_map[np.isnan(corr)] = np.nan
    mean_map[np.isnan(mean)] = np.nan

    np.save(os.path.join(path_results, 'cov.npy'), cov)            
    np.save(os.path.join(path_results, 'corr.npy'), corr)
    np.save(os.path.join(path_results, 'mean.npy'), mean)
    np.save(os.path.join(path_results, 'corr_map.npy'), corr_map)
    np.save(os.path.join(path_results, 'mean_map.npy'), mean_map)

    return corr_map, mean_map

def plot_correlation(corr_map, edt, df_pos, pos, fluo_chns, path_save, path_all, t0):
    #radius = edt.max(axis=(1,2))
    nt, nr, nc =  corr_map.shape
    for i in df_pos.index.values:
        #################################################################################################################
        # Incubation time calculation, should be a function
        time_im = df_pos.loc[i, 't_im']
        time_incub = df_pos.loc[i, 't_incub']
        
        delta_im = timedelta(hours=time_im.hour, minutes=time_im.minute, seconds=time_im.second)
        delta_incub = timedelta(hours=time_incub.hour, minutes=time_incub.minute, seconds=time_incub.second)
        
        # Calculate the difference
        incub_time_s = delta_im - delta_incub
        incub_time_n = incub_time_s.total_seconds() / 60
        
        time_points = np.arange(t0, df_pos.loc[i, 'exp length']) * 10 + incub_time_n
        time_strings = [f"{int(tp // 60)}" for tp in time_points]  # Show only hours

        # Select labels at intervals (e.g., every 50 points)
        interval = 30
        indices = np.arange(0, len(time_points), interval)
        selected_time_strings = [time_strings[j] for j in indices]
        #################################################################################################################
        
        # Plotting
        if fluo_chns == 3:
            plt.figure(figsize=(10, 9))

            ax1 = plt.subplot(3, 1, 1)
            plt.imshow(np.hstack([corr_map[t0:, ::-1, 0], corr_map[t0:, :, 0]]).transpose(), 
                    extent=[0, nt, -edt.max(), edt.max()], 
                    aspect='auto', 
                    cmap='bwr', 
                    vmin=-1, 
                    vmax=1)
            plt.title('Corr(RFP, YFP)')
            plt.colorbar()

            ax2 = plt.subplot(3, 1, 2)
            plt.imshow(np.hstack([corr_map[t0:, ::-1, 1], corr_map[t0:, :, 1]]).transpose(), 
                    extent=[0, nt, -edt.max(), edt.max()],  
                    aspect='auto', 
                    cmap='bwr', 
                    vmin=-1, 
                    vmax=1)
            plt.title('Corr(RFP, CFP)')
            plt.colorbar()
            
            ax3 = plt.subplot(3, 1, 3)
            plt.imshow(np.hstack([corr_map[t0:, ::-1, 2], corr_map[t0:, :, 2]]).transpose(), 
                    extent=[0, nt, -edt.max(), edt.max()],  
                    aspect='auto', 
                    cmap='bwr', 
                    vmin=-1, 
                    vmax=1)
            plt.title('Corr(YFP, CFP)')
            plt.colorbar()

            # Set x-ticks and labels
            for ax in [ax1, ax2, ax3]:
                ax.set_xticks(indices)
                ax.set_xticklabels(selected_time_strings)

            # Add shared x and y labels
            plt.gcf().text(0.5, 0.04, 'Time (h)', ha='center', va='center', fontsize=14)
            plt.gcf().text(0.04, 0.5, r'Radial distance $(\mu m)$', ha='center', va='center', rotation='vertical', fontsize=14)
            
            plt.tight_layout(rect=[0.05, 0.05, 1, 1])  # Adjust layout to make room for the shared labels
            #plt.savefig(os.path.join(path_save,f'pos{pos}',f'corr_pos{pos}.png'))
            plt.savefig(os.path.join(path_all,f'corr_pos{pos}.png'))
            #plt.show()

        elif fluo_chns == 2:
            plt.figure(figsize=(10, 3))
            
            ax1 = plt.subplot(1, 1, 1)
            plt.imshow(np.hstack([corr_map[t0:, ::-1, 0], corr_map[t0:, :, 0]]).transpose(), 
                    extent=[0, nt, -edt.max(), edt.max()], 
                    aspect='auto', 
                    cmap='bwr', 
                    vmin=-1, 
                    vmax=1)
            plt.title('Corr(YFP, CFP)')
            plt.colorbar()

            ax1.set_xticks(indices)
            ax1.set_xticklabels(selected_time_strings)

            # Add shared x and y labels
            plt.gcf().text(0.5, 0.04, 'Time (h)', ha='center', va='center', fontsize=14)
            plt.gcf().text(0.04, 0.5, r'Radial distance $(\mu m)$', ha='center', va='center', rotation='vertical', fontsize=14)
            
            plt.tight_layout(rect=[0.05, 0.05, 1, 1])  # Adjust layout to make room for the shared labels
            #plt.savefig(os.path.join(path_save,f'pos{pos}',f'corr_pos{pos}.png'))
            plt.savefig(os.path.join(path_all,f'corr_pos{pos}.png'))
            #plt.show()

def get_rho_center(im_all, edt, fluo_chns, rfp_chn, yfp_chn, cfp_chn, path_results):
    nt,nx,ny,nc = im_all.shape
    print(im_all.shape)  
    bg = np.zeros((nc,))
    for c in range(nc):
        bg[c] = im_all[0,:100,:100,c].mean()
    
    mean = np.zeros((nt,nc))
    rho = np.zeros((nt,2))
    lrho = np.zeros((nt,2))
    dlrho = np.zeros_like(lrho) + np.nan
    rw = 16
    Rmax = edt.max()

    for t in range(nt):    
        tedt = edt[t,:,:]
        idx = tedt > Rmax - rw        
        if np.sum(idx)>0:
            if fluo_chns == 3:
                ntim0 = im_all[t,:,:,rfp_chn].astype(float) - bg[rfp_chn]
                ntim1 = im_all[t,:,:,yfp_chn].astype(float) - bg[yfp_chn]
                ntim2 = im_all[t,:,:,cfp_chn].astype(float) - bg[cfp_chn]
                x,y,z = ntim0[idx], ntim1[idx], ntim2[idx]
                mean[t,rfp_chn] = x.mean()
                mean[t,yfp_chn] = y.mean()
                mean[t,cfp_chn] = z.mean()            
            elif fluo_chns == 2:                
                ntim0 = im_all[t,:,:,yfp_chn].astype(float) - bg[yfp_chn]
                ntim1 = im_all[t,:,:,cfp_chn].astype(float) - bg[cfp_chn]               
                x,y = ntim0[idx], ntim1[idx]
                mean[t,yfp_chn] = x.mean()
                mean[t,cfp_chn] = y.mean()
    if fluo_chns == 3:
        rho[:,0] = mean[:,0] / mean[:,2]
        rho[:,1] = mean[:,1] / mean[:,2]
        lrho = np.log(rho)
        for c in range(2):
            idx = ~np.isnan(lrho[:,c])
            dlrho[idx,c] = savgol_filter(lrho[idx,c], 21, 3, deriv=1, axis=0) 
    elif fluo_chns == 2:
        rho[:,0] = mean[:,0] / mean[:,1]
        rho[:,1] = mean[:,1] / mean[:,0]
        lrho = np.log(rho)
    for c in range(2):
        idx = ~np.isnan(lrho[:,c])
        dlrho[idx,c] = savgol_filter(lrho[idx,c], 21, 3, deriv=1, axis=0)

    np.save(os.path.join(path_results, 'dlrho_center.npy'), dlrho)
    return dlrho

def fit_velocity(edt, t0, tf, path_results, path_graphs):
    # Fit an exponential decay model to the velocity data
    def residual_func(edt, nvmag, nt, nx, ny):
        def residuals(x):
            r0 = np.exp(x[0])
            C = 0  # x[1]
            res = []
            for frame in range(0,nt):
                for ix in range(nx):
                    for iy in range(ny):
                        if not np.isnan(nvmag[frame, ix, iy]) and vfront[frame] > 1:
                            r = edt[frame, ix * 32:ix * 32 + 64, iy * 32:iy * 32 + 64]
                            #r = edt[t, ix * 16:ix * 16 + 32, iy * 16:iy * 16 + 32]
                            R = rmax[t]
                            B = R / ((R - r0) + r0 * np.exp(-R / r0))    
                            model_vmag = B * (((R - r - r0) * np.exp(-r / r0) + r0 * np.exp(-R / r0)) / (R - r))
                            #B = 1 / (1 - np.exp(-rmax[t] / r0))
                            #model_vmag = 1 + B * (np.exp(-r / r0) - 1)                    
                            mean_model_vmag = vfront[frame] * np.nanmean(model_vmag)
                            res.append(mean_model_vmag - nvmag[frame, ix, iy])
            return res
        return residuals
    
    start_frame = 0
    #t0 = 0
    #tf = 50
    step = 1

    vmag = np.load(os.path.join(path_results, 'vmag.npy'))
    vmag = vmag[t0:tf, :, :]
    nt, nx, ny = vmag.shape
    vfront = np.load(os.path.join(path_results, 'vfront.npy'))
    rmax = np.load(os.path.join(path_results, 'radius.npy'))
    idx = np.arange(start_frame+t0,nt+start_frame+t0)
    vfront = vfront[idx]
    rmax = rmax[idx]

    svmag = np.zeros_like(vmag) + np.nan
    for t in range(0,nt):
        for ix in range(1, nx - 1):
            for iy in range(1, ny - 1):
                svmag[t, ix, iy] = np.nanmean(vmag[t, ix - 1:ix + 2, iy - 1:iy + 2])

    nvmag = np.zeros_like(svmag)
    for frame in range(0,nt):
        #nvmag[frame, :, :] = svmag[frame, :, :] / vfront[frame * step + start_frame]
        nvmag[frame, :, :] = svmag[frame, :, :] / vfront[frame * step]

    radpos = np.load(os.path.join(path_results, 'radpos.npy'))
    radpos = radpos[idx, :, :]
    edt = edt[idx, :, :]
    res = least_squares(residual_func(edt, svmag, nt, nx, ny), x0=(np.log(50),))
    r0 = np.exp(res.x[0])
    C = 0  # res.x[1]
    np.save(os.path.join(path_results, 'r0.npy'), r0)

    print(f'r0 = {r0}, C = {C}')

    ## print results
    # Make a plot to see how good the fit is
    i = 0
    times = np.linspace(0, nt - 1, 12).astype(int)  # [0,int(nt/3),int(2*nt/3),nt-1]
    for t in times:
        plt.subplot(4, 3, i + 1)
        x_model = np.zeros((0,))
        y_model = np.zeros((0,))
        x_data = np.zeros((0,))
        y_data = np.zeros((0,))

        for ix in range(nx):
            for iy in range(ny):
                if not np.isnan(svmag[t, ix, iy]):
                    r = edt[t, ix * 32:ix * 32 + 64, iy * 32:iy * 32 + 64]
                    #r = edt[t, ix * 16:ix * 16 + 32, iy * 16:iy * 16 + 32]
                    x_data = np.append(x_data, np.nanmean(r))
                    y_data = np.append(y_data, svmag[t, ix, iy])
        plt.plot(x_data, y_data, '.', alpha=0.2)  # , color=colours[i])

        for r in np.linspace(0, rmax[t], 100):
            R = rmax[t]
            B = R / ((R - r0) + r0 * np.exp(-R / r0))    
            model_vmag = B * (((R - r - r0) * np.exp(-r / r0) + r0 * np.exp(-R / r0)) / (R - r))

            #B = 1 / (1 - np.exp(-rmax[t] / r0))
            #model_vmag = 1 + B * (np.exp(-r / r0) - 1)
            mean_model_vmag = np.nanmean(model_vmag)
            x_model = np.append(x_model, np.nanmean(r))
            #y_model = np.append(y_model, vfront[start_frame + step * t] * mean_model_vmag)
            y_model = np.append(y_model, vfront[step * t] * mean_model_vmag)

        plt.plot(x_model, y_model, 'k--')  # , color=colours[i])

        i += 1

        plt.title(f't = {idx[t]}')
        plt.xlabel('Radial position')
        plt.ylabel('$v/v_{front}$')
    plt.tight_layout()
    plt.savefig(os.path.join(path_graphs, 'vfront_rad.png'))
    plt.close()

    y_model = np.zeros((0,))
    y_data = np.zeros((0,))
    for t in range(0,nt):
        for ix in range(nx):
            for iy in range(ny):
                if not np.isnan(svmag[t, ix, iy]):
                    r = edt[t, ix * 32:ix * 32 + 64, iy * 32:iy * 32 + 64]
                    #r = edt[t, ix * 16:ix * 16 + 32, iy * 16:iy * 16 + 32]
                    R = rmax[t]
                    B = R / ((R - r0) + r0 * np.exp(-R / r0))    
                    model_vmag = B * (((R - r - r0) * np.exp(-r / r0) + r0 * np.exp(-R / r0)) / (R - r))
                    #B = 1 / (1 - np.exp(-rmax[t] / r0))
                    #model_vmag = 1 + B * (np.exp(-r / r0) - 1)
                    mean_model_vmag = np.nanmean(model_vmag)
                    #y_model = np.append(y_model, vfront[start_frame + step * t] * mean_model_vmag)
                    y_model = np.append(y_model, vfront[step * t] * mean_model_vmag)
                    y_data = np.append(y_data, svmag[t, ix, iy])

    plt.plot(y_model, y_data, '.', alpha=0.1)
    plt.plot([0, y_model.max()], [0, y_model.max()], 'k--')
    plt.xlabel('Model $v$')
    plt.ylabel('Velocimetry $v$')
    plt.savefig(os.path.join(path_graphs, 'model_graph.png'))
    # plt.xscale('log')
    # plt.yscale('log')
    plt.close()

    np.save(os.path.join(path_results, 'x_model.npy'), x_model)
    np.save(os.path.join(path_results, 'x_data.npy'), x_data)
    np.save(os.path.join(path_results, 'y_model.npy'), y_model)
    np.save(os.path.join(path_results, 'y_data.npy'), y_data)
    np.save(os.path.join(path_results, 'svmag.npy'), svmag)

    mu0 = np.zeros((nt,))
    for t in range(t0,nt):

        R = rmax[t]
        B = R / ((R - r0) + r0 * np.exp(-R / r0))
        mu0[t] = vfront[t] / r0 * B

        #B = 1 / (1 - np.exp(-rmax[t] / r0))
        #mu0[t] = 2 * vfront[step * t] / r0 * B

    # The edge growth rate = 2 * edge velocity / r0 
    plt.plot(mu0)
    plt.savefig(os.path.join(path_graphs, 'mu0_profile.png'))
    plt.close()
    np.save(os.path.join(path_results, 'mu0.npy'), mu0)

def get_mean_colony_fluo(im_fluo, edt, fluo_chns, path_results):
    nt, nx ,ny, nc = im_fluo.shape
    bg = np.zeros(nc)
    for c in range(nc):
        bg[c] = im_fluo[0,:100,:100,c].mean(axis=(0,1))

    chns = np.arange(nc)
    mean = np.zeros((nt,nc))
    rho = np.zeros((nt,nc))
    lrho = np.zeros((nt,nc))
    dlrho = np.zeros((nt,nc))
    
    for t in range(nt):
        tedt = edt[t,:,:]
        idx = tedt > 0
        for ch in chns:
            mean[t, ch] = np.nanmean(im_fluo[t, idx, ch] - bg[ch])
    
    if fluo_chns == 2:
        rho[:,0] = mean[:, 0] / mean[:, 1]
        rho[:,1] = mean[:, 1] / mean[:, 0]
        
    elif fluo_chns == 3:
        rho[:,0] = mean[:, 0] / mean[:, 1]
        rho[:,1] = mean[:, 0] / mean[:, 2]
        rho[:,2] = mean[:, 1] / mean[:, 2]
    
    lrho = np.log(rho)
    for c in range(2):
        idx = ~np.isnan(lrho[:,c])
        dlrho[idx,c] = savgol_filter(lrho[idx,c], 21, 3, deriv=1, axis=0)    
    
    np.save(os.path.join(path_results, 'mean_fluo_colony_ts.npy'), mean)
    np.save(os.path.join(path_results, 'mean_fluo_colony.npy'), mean.mean(axis=0))
    np.save(os.path.join(path_results, 'mean_rho_colony_ts.npy'), rho)
    np.save(os.path.join(path_results, 'mean_rho_colony.npy'), rho.mean(axis=0))
    np.save(os.path.join(path_results, 'mean_lrho_colony_ts.npy'), lrho)
    np.save(os.path.join(path_results, 'mean_lrho_colony.npy'), lrho.mean(axis=0))
    np.save(os.path.join(path_results, 'mean_dlrho_colony_ts.npy'), dlrho)
    np.save(os.path.join(path_results, 'mean_dlrho_colony.npy'), dlrho.mean(axis=0))

def get_fluo_edge_center(im_fluo, edt, fluo_chns, rfp_chn, yfp_chn, cfp_chn, path_results):
    nt,nx,ny,nc = im_fluo.shape    
    bg = np.zeros((nc,))
    for c in range(nc):
        bg[c] = im_fluo[0,:100,:100,c].mean()
    
    ## center
    cmean = np.zeros((nt,nc))
    crho = np.zeros((nt,nc))
    clrho = np.zeros((nt,nc))
    cdlrho = np.zeros_like(clrho) + np.nan
    rw = 16
    #Rmax = edt.max()

    ## edge
    emean = np.zeros((nt,nc))
    erho = np.zeros((nt,nc))
    elrho = np.zeros((nt,nc))
    edlrho = np.zeros_like(elrho) + np.nan
    
    for t in range(nt):    
        tedt = edt[t,:,:]
        Rmax = tedt.max()
        ## center
        cidx = tedt > Rmax - rw        
        if np.sum(cidx)>0:
            if fluo_chns == 3:
                cntim0 = im_fluo[t,:,:,rfp_chn].astype(float) - bg[rfp_chn]
                cntim1 = im_fluo[t,:,:,yfp_chn].astype(float) - bg[yfp_chn]
                cntim2 = im_fluo[t,:,:,cfp_chn].astype(float) - bg[cfp_chn]
                x,y,z = cntim0[cidx], cntim1[cidx], cntim2[cidx]
                cmean[t,rfp_chn] = x.mean()
                cmean[t,yfp_chn] = y.mean()
                cmean[t,cfp_chn] = z.mean()            
            elif fluo_chns == 2:                
                cntim0 = im_fluo[t,:,:,yfp_chn].astype(float) - bg[yfp_chn]
                cntim1 = im_fluo[t,:,:,cfp_chn].astype(float) - bg[cfp_chn]               
                x,y = cntim0[cidx], cntim1[cidx]
                cmean[t,yfp_chn] = x.mean()
                cmean[t,cfp_chn] = y.mean()
        # edge
        eidx = (tedt < 2*rw) & (tedt > rw)
        if np.sum(eidx)>0:
            if fluo_chns == 3:
                entim0 = im_fluo[t,:,:,rfp_chn].astype(float) - bg[rfp_chn]
                entim1 = im_fluo[t,:,:,yfp_chn].astype(float) - bg[yfp_chn]
                entim2 = im_fluo[t,:,:,cfp_chn].astype(float) - bg[cfp_chn]
                x,y,z = entim0[eidx], entim1[eidx], entim2[eidx]
                emean[t,rfp_chn] = x.mean()
                emean[t,yfp_chn] = y.mean()
                emean[t,cfp_chn] = z.mean()            
            elif fluo_chns == 2:                
                entim0 = im_fluo[t,:,:,yfp_chn].astype(float) - bg[yfp_chn]
                entim1 = im_fluo[t,:,:,cfp_chn].astype(float) - bg[cfp_chn]               
                x,y = entim0[eidx], entim1[eidx]
                emean[t,yfp_chn] = x.mean()
                emean[t,cfp_chn] = y.mean()
    # center
    if fluo_chns == 3:
        crho[:,0] = cmean[:,0] / cmean[:,1] # ry
        crho[:,1] = cmean[:,0] / cmean[:,2] # rc
        crho[:,2] = cmean[:,1] / cmean[:,2] # yc
        clrho = np.log(crho)
     
    elif fluo_chns == 2:
        crho[:,0] = cmean[:,0] / cmean[:,1] # yc
        crho[:,1] = cmean[:,1] / cmean[:,0] # cy
        clrho = np.log(crho)
    for c in range(2):
        idx = ~np.isnan(clrho[:,c])
        cdlrho[idx,c] = savgol_filter(clrho[idx,c], 21, 3, deriv=1, axis=0)
    # edge
    if fluo_chns == 3:
        erho[:,0] = emean[:,0] / emean[:,1] # ry
        erho[:,1] = emean[:,0] / emean[:,2] # rc
        erho[:,2] = emean[:,1] / emean[:,2] # yc
        elrho = np.log(erho)
    elif fluo_chns == 2:
        erho[:,0] = emean[:,0] / emean[:,1] # yc
        erho[:,1] = emean[:,1] / emean[:,0] # cy
        elrho = np.log(erho)
    for c in range(nc):
        idx = ~np.isnan(elrho[:,c])
        edlrho[idx,c] = savgol_filter(elrho[idx,c], 21, 3, deriv=1, axis=0)
    np.save(os.path.join(path_results, 'cmean.npy'), cmean)
    np.save(os.path.join(path_results, 'crho.npy'), crho)
    np.save(os.path.join(path_results, 'clrho.npy'), clrho)
    np.save(os.path.join(path_results, 'cdlrho.npy'), cdlrho)
    np.save(os.path.join(path_results, 'emean.npy'), emean)
    np.save(os.path.join(path_results, 'erho.npy'), erho)              
    np.save(os.path.join(path_results, 'elrho.npy'), elrho)
    np.save(os.path.join(path_results, 'edlrho.npy'), edlrho)