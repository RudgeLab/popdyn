import os
import json
from datetime import datetime
import math
import numpy as np
import pandas as pd
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

def get_frames_vel(res):
    i = res.index[0]
    incub_time_s = datetime.strptime(res.loc[i,'t_im'], '%H:%M:%S') - datetime.strptime(res.loc[i,'t_incub'], '%H:%M:%S')
    incub_time_n = incub_time_s.seconds / 60
    t_m = res.loc[i, 't_m']
    params = list(json.loads(res.loc[i,'gomp_params']).values())
    
    tmax = 4 * np.log(2) / params[2]
    fini = math.ceil((t_m - incub_time_n)/10)
    fend = 3 + math.ceil(tmax/10)
    return fini, fend
        #print(f"Frame ini: {fini}, Frame fin: {fend}")
        #print(f"Frames: {fend - fini}")

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


##################
# global params
##################

# experiments metadata
with open('metadata.json') as f:
    metadata = json.load(f)

##################
# analysis params
##################

#folder = '/home/guillermo/Microscopy'
#folder = '/mnt/ff9e5a34-3696-46e4-8fa8-0171539135be'
path_ext = '/media/c1046372/Expansion/Thesis GY/3. Analyzed files/'
#scope_name = 'Ti scope'
#scope_name = 'Tweez scope'
#path_scope = os.path.join(folder, scope_name)
#exp_date = '2023_12_08'
#df_date = '2023-11-17'
#path = os.path.join(path_scope, exp_date)
folder_masks = 'contour_masks'
folder_results = 'results'
folder_fluo = 'fluo'
folder_graphs = 'graphs'
folder_velocity = 'velocity_data'
# for file name
scopes = {'Tweez scope': 'TiTweez', 'Ti scope': 'Ti'}
dnas = {'pLPT20&pLPT41': 'pLPT20&41', 'pLPT119&pLPT41': 'pLPT119&41', 'pAAA': 'pAAA', 'pLPT107&pLPT41': 'pLPT107&41'}
#vector = 'pLPT20&pLPT41'

#df = pd.read_excel('Notebooks/out_gomp_log.xlsx')

# call get_params_for_date and then make a loop

# experiments metadata
#with open('metadata.json') as f:
#    metadata = json.load(f)

# this is for bulk analysis
exp_sum = pd.read_excel('../Notebooks/Exps_summary.xlsx')
exp_sum['formatted_dates'] = exp_sum['Date'].dt.strftime('%Y_%m_%d')
positions = pd.read_excel('../Notebooks/Positions.xlsx')

for i in range(len(exp_sum)):
    exp_date = exp_sum.loc[i,'formatted_dates']
    vector = exp_sum.loc[i,'DNA']

    scope_name = exp_sum.loc[i,'Machine']
    poss = positions[(positions.Date == exp_sum.loc[i, 'Date']) & (positions.DNA == vector) & (positions.Machine == scope_name) & (positions.Quality == 'Very good')].Position.unique()

    if vector == 'pLPT20&pLPT41' or vector == 'pLPT119&pLPT41':
        yfp_chn = 0
        cfp_chn = 1
        ph_chn = 2
        fluo_chns = 2
    else:
        rfp_chn = 0
        yfp_chn = 1
        cfp_chn = 2
        ph_chn = 3
        fluo_chns = 3
    # loop to perform the functions contour_mask, average_growth, compute_er to each
    # position (colony) selected from an experiment
    for pos in poss:
    #for pos in [38]: #[23,24,25,28,30,31,32,34]:
        print(f"Pos {pos}")
        print(f"{exp_date}_{scopes[scope_name]}_{vector}")
        fname = f'{exp_date}_10x_1.0x_{dnas[vector]}_{scopes[scope_name]}_Pos{pos}.ome.tif'
        path_scope = os.path.join(path_ext, scope_name)
        path = os.path.join(path_scope, exp_date)
        path_im = os.path.join(path, fname)
        path_results = os.path.join(path, folder_results, f"pos{pos}")
        path_graphs = os.path.join(path, folder_graphs)       
        fname_mask = 'mask_' + fname
        path_masks = os.path.join(path, folder_masks, fname_mask)
        colonies = get_params_for_date(scope_name, exp_date, metadata)

        
        # create folders that will store analysis results
        if not os.path.exists(os.path.join(path, folder_masks)):
            os.makedirs(os.path.join(path, folder_masks))
        if not os.path.exists(os.path.join(path, folder_results)):
            os.makedirs(os.path.join(path, folder_results))
        if not os.path.exists(os.path.join(path, folder_fluo)):
            os.makedirs(os.path.join(path, folder_fluo))
        if not os.path.exists(os.path.join(path, folder_graphs)):
            os.makedirs(os.path.join(path, folder_graphs))
        if not os.path.exists(os.path.join(path, folder_velocity)):
            os.makedirs(os.path.join(path, folder_velocity))
        
        im_all = imread(path_im)
        im_ph = im_all[:,:,:,ph_chn].astype(float)
        im_yfp = im_all[:,:,:,yfp_chn].astype(float)
        im_fluo = im_all[:,:,:,:fluo_chns].astype(float)
        
        ###############
        # contour mask
        # it's in the inverse order because x -> columns and y -> rows
        #cy = metadata[scope_name][exp_date][str(pos)]['cx']
        #cx = metadata[scope_name][exp_date][str(pos)]['cy']
        # TO DO: 'radius' is the guest to start the segmentation, change this name
        #radius = metadata[scope_name][exp_date][str(pos)]['radius']
        #radj = metadata[scope_name][exp_date][str(pos)]['radj']
        
        

        #contour_mask(im_ph, start_frame, step, pos, cx, cy, radius, path, folder_masks, path_masks, radj)
        
        ###############
        # average_growth
        #average_growth(path_masks, step, pos, path, folder_results, folder_graphs)

        ####################
        # velocity profile
        ####################
        
        #fini = metadata[scope_name][exp_date][str(pos)]['vini']
        #nframes = metadata[scope_name][exp_date][str(pos)]['vfin']
        #windowsize = metadata[scope_name][exp_date][str(pos)]['wsize']
        #windowspacing = metadata[scope_name][exp_date][str(pos)]['wspacing']
        startframe = 0
        step = 1
        nframes = 70
        windowsize = 64
        windowspacing = 32
        
        #print(fini, fend)
        compute_velocity(startframe, nframes, im_ph, path_masks, path, folder_velocity, pos, windowsize, windowspacing)
        ###############
        # compute_er
        #er, edt_reg, sfluo, dsfluo = compute_er(im_all, pos, path, folder_results, fname, ph_chn)

        ###############
        # plot expression rate
        #plot_er(im_ph, pos, path, folder_fluo, er, edt_reg, sfluo, dsfluo, fluo_chns)

        ###############
        """
        # videos expression rate
        vids = ["sfluo", "dsfluo", "er"]
        for i in range(len(vids)):
            path_ims = os.path.join(path, folder_fluo, f"pos{pos}", vids[i])
            path_out = os.path.join(path, folder_results, f"pos{pos}_{vids[i]}.avi")
            print(f"path_ims: {path_ims}")
            print(f"path_out: {path_out}")
            make_video(path_ims, path_out)
        """
