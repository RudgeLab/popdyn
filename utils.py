import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

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

# función para hacer el video: toma archivos .png y construye un video
# recibe como parámetros image_folder (carpeta donde están los archivos .png guardados) 
#y video_name (nombre archivo de video)
def make_video(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 7, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()

def contour_mask(im_ph, start_frame, step, cx, cy, radius):
    nt,nx,ny = im_ph.shape
    mask_out = np.zeros((nt,) + (nx,ny))

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
        plt.savefig(os.path.join(path, folder_masks,'contour_%03d.png'%t))
        plt.close()