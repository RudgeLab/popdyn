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

def compute_corr(im_all, edt_path, nr, rw, rfp_chn, yfp_chn, cfp_chn):    
    nt,nx,ny,nc = im_all.shape
    rs = np.linspace(rw, edt.max(), nr)
    bg = np.zeros((nc,))
    for c in range(nc):
        bg[c] = im_all[0,:100,:100,c].mean()

    edt = np.load(os.path.join(edt_path))
    edt = edt[:,:,:]

    cov = np.zeros((nt,nr,nc,nc))
    corr = np.zeros((nt,nr,nc))
    mean = np.zeros((nt,nr,nc))
    for t in range(nt):
        for ri in range(nr):
            tedt = edt[t,:,:]
            idx = np.abs(tedt - rs[ri]) < rw
            if np.sum(idx)>0:
                #plt.figure()
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
    np.save(os.path.join(path_results, 'cov.npy'), cov)            
    np.save(os.path.join(path_results, 'corr.npy'), corr)
    np.save(os.path.join(path_results, 'mean.npy'), mean)

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
    poss = positions[(positions.Date == exp_sum.loc[i, 'Date']) & 
                     (positions.DNA == vector) & 
                     (positions.Machine == scope_name) & 
                     (positions.Quality == 'Very good')].Position.unique()

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

        start_frame = 0
        step = 1
        nframes = 70
        windowsize = 64
        windowspacing = 32
        
        #print(fini, fend)
        #compute_velocity(start_frame, nframes, im_ph, path_masks, path, folder_velocity, pos, windowsize, windowspacing)
        ###############
        
        # process velocity
        process_velocity(path, fname, folder_velocity, folder_results, folder_masks, pos, start_frame, step)
        
        edt_path = os.path.join(path_results,'edt.npy')
        pad = 32
        nr = 64
        rw = 8
        #rw = 16
        #rs = np.linspace(rw, edt.max(), nr)

        # compute_corr and mean
        # image is just fluo channels
        compute_corr(im_fluo, edt_path, nr, rw, rfp_chn, yfp_chn, cfp_chn) 


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
