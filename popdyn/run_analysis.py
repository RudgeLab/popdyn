import os
import json
import gc
from datetime import datetime, timedelta
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

##################
# global params
##################

# experiments metadata
with open('metadata.json') as f:
    metadata = json.load(f)

##################
# analysis params
##################

#path_ext = '/home/guillermo/Microscopy'
#path_ext = '/mnt/ff9e5a34-3696-46e4-8fa8-0171539135be'
#path_ext = '/media/c1046372/Expansion/Thesis GY/3. Analyzed files/'
path_ext = '/media/guillermo/Expansion/Thesis GY/3. Analyzed files'
#exp_date = '2023_11_15'
#vector = 'pLPT20&pLPT41'
#scope_name = 'Ti scope'
#scope_name = 'Tweez scope'
#path_scope = os.path.join(folder, scope_name)
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

# experiments metadata
#with open('metadata.json') as f:
#    metadata = json.load(f)

# this is for bulk analysis
exp_sum = pd.read_excel('../Notebooks/Exps_summary.xlsx')
exp_sum['formatted_dates'] = exp_sum['Date'].dt.strftime('%Y_%m_%d')
positions = pd.read_excel('../Notebooks/Positions.xlsx')

df = pd.read_excel('../Notebooks/data_processed.xlsx')

for i in [0]:#exp_sum.index.values:
    exp_date = exp_sum.loc[i,'formatted_dates']
    vector = exp_sum.loc[i,'DNA']
    scope_name = exp_sum.loc[i,'Machine']
    df_pos = positions[(positions.Date == exp_sum.loc[i, 'Date']) & 
        (positions.DNA == vector) & 
        (positions.Machine == scope_name) & 
        (positions.Quality == 'Very good')]
    poss = df_pos.Position.unique()
    
    if vector == 'pLPT20&pLPT41' or vector == 'pLPT119&pLPT41':
        rfp_chn = ''
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
    for pos in [7]:#poss:
    #for pos in [14,15,16]:
        t0 = positions[(positions.Date == exp_sum.loc[i, 'Date']) & 
        (positions.DNA == vector) & 
        (positions.Machine == scope_name) & 
        (positions.Quality == 'Very good')& 
        (positions.Position == pos)]

        print(f"Pos {pos}")
        print(f"{exp_date}_{scopes[scope_name]}_{vector}")
        fname = f'{exp_date}_10x_1.0x_{dnas[vector]}_{scopes[scope_name]}_Pos{pos}.ome.tif'
        path_scope = os.path.join(path_ext, scope_name)
        path = os.path.join(path_scope, exp_date)
        path_im = os.path.join(path, fname)
        path_results = os.path.join(path, folder_results, f"pos{pos}")
        path_graphs = os.path.join(path, folder_graphs, f"pos{pos}") 
        path_velocity =   os.path.join(path, folder_velocity, f"pos{pos}")     
        fname_mask = 'mask_' + fname
        path_masks = os.path.join(path, folder_masks, fname_mask)
        
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
        #im_yfp = im_all[:,:,:,yfp_chn].astype(float)
        #im_fluo = im_all[:,:,:,:fluo_chns].astype(float)
        edt_path = os.path.join(path_results,'edt.npy')
        edt = np.load(os.path.join(edt_path))
        #edt = edt[:,:,:]
        start_frame = 0
        step = 1

        ###############
        # Contour mask
        ###############

        # it's in the inverse order because x -> columns and y -> rows
        cy = metadata[scope_name][exp_date][str(pos)]['cx']
        cx = metadata[scope_name][exp_date][str(pos)]['cy']
        # TO DO: 'radius' is the guest to start the segmentation, change this name
        radius = metadata[scope_name][exp_date][str(pos)]['radius']
        radj = metadata[scope_name][exp_date][str(pos)]['radj']
        #t0 = metadata[scope_name][exp_date][str(pos)]['vini']
        #tf = metadata[scope_name][exp_date][str(pos)]['vend']
        #contour_mask(im_ph, start_frame, step, pos, cx, cy, radius, path, folder_masks, path_masks, radj)
        
        #################
        # average_growth
        #################
        average_growth(path_masks, step, pos, path, folder_results, folder_graphs)

        ####################
        # Velocity profile
        ####################
        #start_frame = 0
        #step = 1        
        nframes = 70
        windowsize = 64
        windowspacing = 32
        
        #compute_velocity(start_frame, nframes, im_ph, path_masks, path, folder_velocity, pos, windowsize, windowspacing)
        ###############
        
        # process velocity
        #process_velocity(path, fname, folder_velocity, folder_results, folder_masks, pos, start_frame, step)
        """
        t0 = int(df[(df['Date'].dt.strftime('%Y_%m_%d') == exp_date) & 
            (df.Machine == scope_name) & 
            (df.Position == pos)]['vel_fit_start'].values[0])

        tf = int(df[(df['Date'].dt.strftime('%Y_%m_%d') == exp_date) & 
            (df.Machine == scope_name) & 
            (df.Position == pos)]['vel_fit_end'].values[0])
        fit_velocity(edt, t0, tf, path_results, path_graphs)
        """
        
        ################
        # Kymo
        ################
        nr = 64
        rw = 16
        rs = np.linspace(rw, edt.max(), nr)

        #kymo = compute_kymo(im_fluo, edt, nr, rw)
        #np.save(os.path.join(path_results, 'kymo.npy'), kymo)
        #dlkymo = compute_dlkymo(kymo, nr, fluo_chns)
        #np.save(os.path.join(path_results, 'dlrho.npy'), dlkymo)
        #plot_fluo_ratio(dlkymo, edt, path, rs, pos, fluo_chns)
          
        ####################
        # Correlation
        ####################

        pad = 32
        nr = 64
        rw = 16
        # compute_corr and mean
        # image is just fluo channels
        #corr_map, mean_map = compute_corr(im_fluo, edt, nr, rw, rfp_chn, yfp_chn, cfp_chn) 
        #corr_map, _ = compute_corr(im_fluo, edt, nr, rw, fluo_chns, rfp_chn, yfp_chn, cfp_chn, path_results) 

        # plot corr
        #t0 = 50
        #path_all = os.path.join(path, folder_graphs)
        #plot_correlation(corr_map, edt, df_pos, pos, fluo_chns, path_graphs, path_all, t0)

        # add return later
        # this function gets mean, rho, lrho and dlrho in time and for the whole TL in a single position
        #get_mean_colony_fluo(im_fluo, edt, fluo_chns, path_results)
        #get_fluo_edge_center(im_fluo, edt, fluo_chns, rfp_chn, yfp_chn, cfp_chn, path_results)
        #_ = get_rho_center(im_fluo, edt, fluo_chns, rfp_chn, yfp_chn, cfp_chn, path_results)
        ### Some cleaning
        del edt
        #del corr_map
        del im_all
        del im_ph
        #del im_fluo
        #del mean_map
        #del kymo
        #del dlkymo
        # compute_er
        #er, edt_reg, sfluo, dsfluo = compute_er(im_all, pos, path, folder_results, fname, ph_chn)

        ###############
        # plot expression rate
        #plot_er(im_ph, pos, path, folder_fluo, er, edt_reg, sfluo, dsfluo, fluo_chns)

        ###############
        gc.collect()
