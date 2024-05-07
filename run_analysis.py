import os
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave

from scipy.ndimage import distance_transform_edt
from scipy.optimize import fmin, least_squares
from scipy.signal import savgol_filter

from skimage.measure import find_contours
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.morphology import binary_erosion
from skimage.draw import polygon

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
with open('metadata.json') as f:
    metadata = json.load(f)

##################
# analysis params
##################

folder = '/home/guillermo/Microscopy'
#folder = '/mnt/ff9e5a34-3696-46e4-8fa8-0171539135be'
scope_name = 'Ti scope'
#scope_name = 'Tweez scope'
path_scope = os.path.join(folder, scope_name)
exp_date = '2023_12_08'
path = os.path.join(path_scope, exp_date)
folder_masks = 'contour_masks'
folder_results = 'results'
folder_fluo = 'fluo'
folder_graphs = 'graphs'

### params for repressilator single reporter
"""
yfp_chn = 0
cfp_chn = 1
ph_chn = 2
fluo_chns = 2
"""
### params for repressilator triple reporter and pAAA
#"""
rfp_chn = 0
yfp_chn = 1
cfp_chn = 2
ph_chn = 3
fluo_chns = 3
#"""

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
for pos in [23]: #[23,24,25,28,30,31,32,34]:
    # TO DO: fname needs to be more modular
    #fname = f'{exp_date}_10x_1.0x_pLPT20&41_TiTweez_Pos{pos}.ome.tif'
    #fname = f'{exp_date}_10x_1.0x_pLPT20&41_Ti_Pos{pos}.ome.tif'
    #fname = f'{exp_date}_10x_1.0x_pLPT119&41_Ti_Pos{pos}.ome.tif'
    #fname = f'{exp_date}_10x_1.0x_pLPT119&41_TiTweez_Pos{pos}.ome.tif'
    fname = f'{exp_date}_10x_1.0x_pLPT107&41_Ti_Pos{pos}.ome.tif'
    #fname = f'{exp_date}_10x_1.0x_pLPT20&41_TiTweez_Pos{pos}.ome.tif'
    #fname = f'{exp_date}_10x_1.0x_pLPT107&41_TiTweez_Pos{pos}.ome.tif'
    #fname = f'{exp_date}_10x_1.0x_pAAA_TiTweez_Pos{pos}.ome.tif'
    #fname = f'{exp_date}_10x_1.0x_pAAA_Ti_Pos{pos}.ome.tif'
    fname_mask = 'mask_' + fname

    path_im = os.path.join(path, fname)
    path_masks = os.path.join(path, folder_masks, fname_mask)
    
    start_frame = 0
    step = 1

    # TO DO: make this parametric
    # for experiments with more than 216 frames
    #im_all = imread(path_im)[:315,:,:,:]
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
    radj = metadata[scope_name][exp_date][str(pos)]['radj']

    #contour_mask(im_ph, start_frame, step, pos, cx, cy, radius, path, folder_masks, path_masks, radj)
    
    ###############
    # average_growth
    average_growth(path_masks, step, pos, path, folder_results, folder_graphs)

    ###############
    # compute_er
    er, edt_reg, sfluo, dsfluo = compute_er(im_all, pos, path, folder_results, fname, ph_chn)

    ###############
    # plot expression rate
    plot_er(im_ph, pos, path, folder_fluo, er, edt_reg, sfluo, dsfluo, fluo_chns)

    ###############
    #"""
    # videos expression rate
    vids = ["sfluo", "dsfluo", "er"]
    for i in range(len(vids)):
        path_ims = os.path.join(path, folder_fluo, f"pos{pos}", vids[i])
        path_out = os.path.join(path, folder_results, f"pos{pos}_{vids[i]}.avi")
        print(f"path_ims: {path_ims}")
        print(f"path_out: {path_out}")
        make_video(path_ims, path_out)
    #"""