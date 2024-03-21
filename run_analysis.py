import os
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.optimize import fmin, least_squares
from scipy.signal import savgol_filter

from skimage.measure import find_contours
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.morphology import binary_erosion
from skimage.draw import polygon
import matplotlib.pyplot as plt

from utils import *

#########################
# Functions
#########################
def get_params_for_date(microscope, date, metadata):
    """Retrieve all parameters for a given microscope and date."""
    try:
        date_data = metadata[microscope][date]
        for pos, data in date_data.items():
            print(f"{microscope} for {date}: Pos{pos}, {data}")
    except KeyError:
        print("Data not found for the given microscope and date.")

##################
# global params
##################

# experiments metadata
# TO DO: store it in a file
metadata = {
    'Tweez scope': {
        '2023_11_15': {
            '0': {'cx': 524, 'cy': 480, 'radius': 100},
            '1': {'cx': 505, 'cy': 520, 'radius': 100},
            '3': {'cx': 510, 'cy': 510, 'radius': 100},
            # More positions
        },
        # More dates
    },
    'Ti scope': {
        '2023_11_15': {
            '0': {'cx': 600, 'cy': 600, 'radius': 100},
            '1': {'cx': 600, 'cy': 600, 'radius': 100},
            '2': {'cx': 600, 'cy': 600, 'radius': 100},
            '3': {'cx': 600, 'cy': 600, 'radius': 100},
            '6': {'cx': 625, 'cy': 590, 'radius': 110},
            '7': {'cx': 600, 'cy': 580, 'radius': 100},
            '9': {'cx': 605, 'cy': 590, 'radius': 100},
            '17': {'cx': 610, 'cy': 590, 'radius': 100},
            '19': {'cx': 600, 'cy': 600, 'radius': 100},
            # More positions
        },
        # More dates
    }
}

##################
# analysis params
##################

#folder = '/home/guillermo/Microscopy/Ti scope'
folder = '/mnt/ff9e5a34-3696-46e4-8fa8-0171539135be/Tweez scope'
exp_date = '2023_11_28'
path = os.path.join(folder, exp_date)
folder_masks = 'contour_masks'
folder_fluo = 'fluo'
folder_graphs = 'graphs'

pos = 1
fname = f'2023_11_28_10x_1.0x_pAAA_TiTweez_Pos{pos}.ome.tif'
fname_mask = 'mask_' + fname

path_im = os.path.join(path, fname)
path_masks = os.path.join(path, folder_masks, fname_mask)


###############
# contour mask
###############
start_frame = 0
step = 1

im_all = imread(path_im)
im_ph = im_all[:,:,:,3]
im_ph = im_ph.astype(float)

nt,nx,ny = im_ph.shape
# input manually from image inspection

cx,cy = 520,520
radius = 100
