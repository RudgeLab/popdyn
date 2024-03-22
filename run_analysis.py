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
# this function might be needed in future when the metadata file gets bigger
def get_params_for_date(microscope, date, metadata):
    """Retrieves all parameters for a given microscope and date.
    
    Parameters:
    - microscope (string): Name of the microscope that performed the experiment.
    - date (string): Date of the experimental data to be analyzed.
    - metadata (dict): Dictionary that contains experimental metadata

    Returns:
    - dictionary: Information of the colonies selected for analysis: Pos, cx, cy, radius.
    """
    try:
        return metadata[microscope][date]
    except KeyError:
        print("Data not found for the given microscope and date.")

##################
# global params
##################

# experiments metadata
# TO DO: store in a file
metadata = {
    'Tweez scope': {
        '2023_11_15': {
            #'0': {'cx': 524, 'cy': 480, 'radius': 100}, #originally taken the other way around, (x, y) instead of (row, col)
            '0': {'cx': 480, 'cy': 524, 'radius': 100},
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

#folder = '/home/guillermo/Microscopy'
folder = '/mnt/ff9e5a34-3696-46e4-8fa8-0171539135be'
#scope_name = 'Ti scope'
scope_name = 'Tweez scope'
path_scope = os.path.join(folder, scope_name)
exp_date = '2023_11_15'
path = os.path.join(path_scope, exp_date)
folder_masks = 'contour_masks'
folder_results = 'results'
folder_fluo = 'fluo'
folder_graphs = 'graphs'

### params for repressilator single reporter
yfp_chn = 0
cfp_chn = 1
ph_chn = 2
fluo_chns = 2

### params for repressilator triple reporter and pAAA
"""
rfp_chn = 0
yfp_chn = 1
cfp_chn = 2
ph_chn = 3
fluo_chns = 3
"""

# create folders that will store analysis results
if not os.path.exists(os.path.join(path, folder_masks)):
    os.makedirs(os.path.join(path, folder_masks))
if not os.path.exists(os.path.join(path, folder_results)):
    os.makedirs(os.path.join(path, folder_results))
if not os.path.exists(os.path.join(path, folder_fluo)):
    os.makedirs(os.path.join(path, folder_fluo))
if not os.path.exists(os.path.join(path, folder_graphs)):
    os.makedirs(os.path.join(path, folder_graphs))

# call get_params_for_date and then make a loop
colonies = get_params_for_date(scope_name, exp_date, metadata)

# loop to perform the functions contour_mask, average_growth, compute_er to each
# position (colony) selected from an experiment
#for pos in colonies.keys():
for pos in range(1):
    # TO DO: fname needs to be more modular
    fname = f'{exp_date}_10x_1.0x_pLPT20&41_single_TiTweez_Pos{pos}.ome.tif'
    fname_mask = 'mask_' + fname

    path_im = os.path.join(path, fname)
    path_masks = os.path.join(path, folder_masks, fname_mask)
    
    start_frame = 0
    step = 1

    im_all = imread(path_im)
    im_ph = im_all[:,:,:,ph_chn].astype(float)
    im_fluo = im_all[:,:,:,:fluo_chns].astype(float)
    
    ###############
    # contour mask
    # it's in the inverse order because x -> columns and y -> rows
    cy = metadata[scope_name][exp_date][str(pos)]['cx']
    cx = metadata[scope_name][exp_date][str(pos)]['cy']
    # TO DO: 'radius' is the guest to start the segmentation, change this name
    radius = metadata[scope_name][exp_date][str(pos)]['radius']

    #contour_mask(im_ph, start_frame, step, pos, cx, cy, radius, path, folder_masks, path_masks)
    
    ###############
    # average_growth
    # average_growth(path_masks, step, pos, path, folder_results, folder_graphs)

    ###############
    # compute_er
    # compute_er(im_all, pos, path, folder_results, ph_chn)

    ###############
    # plot expression rate
    folder_pos = os.path.join(path, folder_results, f"pos{pos}")
    er = np.load(os.path.join(folder_pos, 'er.npy'))
    edt = np.load(os.path.join(folder_pos, 'edt.npy'))
    sfluo = np.load(os.path.join(folder_pos, 'sfluo.npy'))
    dsfluo = np.load(os.path.join(folder_pos, 'dsfluo.npy'))
    nt, nx, ny, nc = sfluo.shape
    plot_er(im_ph, pos, path, folder_fluo, er, edt, sfluo, dsfluo)