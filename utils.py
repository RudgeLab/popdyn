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

from scipy.ndimage import distance_transform_edt
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
        for c in range(nc):
            im[t,:,:,c] = im[t,:,:,c] - bg[c]
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

def average_growth(im_fluo, path_masks, step, pos, path, folder_results, folder_graphs):
    
    folder_pos = os.path.join(path, folder_results,f"pos{pos}")
    if not os.path.exists(folder_pos):
        os.makedirs(folder_pos)

    folder_pos_graph = os.path.join(path, folder_graphs,f"pos{pos}")
    if not os.path.exists(folder_pos_graph):
        os.makedirs(folder_pos_graph)
    
    # BG correction using first frame, too much scattering in later frames
    im_fluo = bg_corr(im_fluo, 0, 100, 0, 100)

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

    np.save(os.path.join(path, folder_results, folder_pos, 'radius.npy'), radius)
    np.save(os.path.join(path, folder_results, folder_pos, 'area.npy'), area) 
    np.save(os.path.join(path, folder_results, folder_pos, 'vfront.npy'), vfront)
    np.save(os.path.join(path, folder_results, folder_pos, 'expansion_rate.npy'), exprate)
    np.save(os.path.join(path, folder_results, folder_pos, 'edt.npy'),  edt)

    plt.subplot(4,1,1)
    plt.plot(area)
    plt.subplot(4,1,2)
    plt.plot(radius)
    plt.subplot(4,1,3)
    plt.plot(vfront)
    plt.subplot(4,1,4)
    plt.plot(exprate)
    plt.savefig(os.path.join(path, folder_results, folder_pos_graph, 'a_r_vf_exp.png'))
    plt.close()