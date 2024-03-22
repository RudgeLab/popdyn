import os
import numpy as np
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

def make_video(images_folder, video_name):
    """
    Makes a video from a sequence of png images.
    
    Parameters:
    - images_folder (string): Folder that contains the png files.
    - video_name (string): Date of the experimental data to be analyzed.

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

def contour_mask(im_ph, start_frame, step, pos, cx, cy, radius, path, folder_masks, path_masks):
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
        plt.savefig(os.path.join(path, folder_masks,f"temp_pos{pos}",'contour_%03d.png'%t))
        plt.close()

    imsave(path_masks, mask_out>0)
    make_video(os.path.join(path, folder_masks,f"temp_pos{pos}"), 
           os.path.join(path, folder_masks,f"contour_pos{pos}.avi"))

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

def compute_er(im, pos, path, folder_results, ph_chn):
    folder_pos = os.path.join(path, folder_results, f"pos{pos}")
    nt, nx, ny, nc = im.shape
    tmin = 0
    tmax = nt

    area = np.load(os.path.join(folder_pos, 'area.npy'))
    radius = np.sqrt(area/np.pi)
    # TO DO: not used nor saved
    dsarea = savgol_filter(area, 21, 3, deriv=1)
    sradius = savgol_filter(radius, 21, 3)
    dsradius = savgol_filter(radius, 21, 3, deriv=1)

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
    # save im and edt registered
    np.save(os.path.join(folder_pos, 'im_reg.npy'), im)
    np.save(os.path.join(folder_pos, 'edt_reg.npy'), edt)
    
    # test
    #imsave(os.path.join(path, folder_results, folder_pos, 'im_reg.ome.tif'), im)

    # select a ROI of the image to analyze
    y,x = np.meshgrid(np.arange(ny), np.arange(nx))
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

    fluo = fluo[:,xmin:xmax,ymin:ymax,:]
    nt,nx,ny,nc = fluo.shape

    sfluo = savgol_filter(fluo, 31, 3, axis=0)
    dsfluo = savgol_filter(fluo, 31, 3, deriv=1, axis=0)
    np.save(os.path.join(folder_pos, 'sfluo.npy'), sfluo)
    np.save(os.path.join(folder_pos, 'dsfluo.npy'), dsfluo)
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

def plot_er(im_ph, pos, path, folder_fluo, er, edt, sfluo, dsfluo):
    folder_pos = os.path.join(path, folder_fluo, f"pos{pos}")
    if not os.path.exists(folder_pos):
        os.makedirs(folder_pos)    

    vmin = [0]*3
    vmax = np.nanmax(er, axis=(0,1,2))

    nt, nx, ny = edt.shape
    y,x = np.meshgrid(np.arange(ny), np.arange(nx))
    edt0 = edt[-1,:,:]
    xmin = x[edt0>0].min() - 32
    xmax = x[edt0>0].max() + 32
    ymin = y[edt0>0].min() - 32
    ymax = y[edt0>0].max() + 32

    # TO DO: make edt.shape and ph.shape match with sfluo.shape 
    edt = edt[:,xmin:(xmax+1),ymin:(ymax+1)]
    ph = im_ph[:,xmin:(xmax+1),ymin:(ymax+1)]
    #############################################

    nt, nx, ny = edt.shape
    _, _, _, nc = sfluo.shape

    print(f"edt: {edt.shape}")
    print(f"sfluo: {sfluo.shape}")
    print(f"ph: {ph.shape}")

    for t in range(nt):
        print(f"Plotting dsfluo {t+1} / {nt}")
        plt.figure(figsize=(9,3))
        for c in range(nc):
            plt.subplot(1,nc,c+1)
            tcdsfluo = dsfluo[t,:,:,c]
            tcdsfluo[edt[t,:,:]==0] = np.nan
            plt.imshow(tcdsfluo)
            plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(folder_pos, 'dsfluo_%04d.png'%t))
        plt.close()

    for t in range(nt):
        print(f"Plotting sfluo {t+1} / {nt}")
        plt.figure(figsize=(9,3))
        for c in range(nc):
            plt.subplot(1,nc,c+1)
            tcsfluo = sfluo[t,:,:,c]
            tcsfluo[edt[t,:,:]==0] = np.nan
            plt.imshow(tcsfluo)
            plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(folder_pos, 'sfluo_%04d.png'%t))
        plt.close()

    for t in range(nt):
        print(f"Plotting er {t+1} / {nt}")
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
        plt.savefig(os.path.join(folder_pos, 'er_%04d.png'%t))
        plt.close()