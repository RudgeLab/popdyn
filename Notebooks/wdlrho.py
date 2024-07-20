import os
import json
import re
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from skimage.io import imread, imsave
from skimage.transform import warp
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
import scipy.stats as stats
from scipy.integrate import odeint, solve_ivp
from numba import njit, prange
from skimage.filters import gaussian
from scipy.ndimage import uniform_filter1d
from numpy.polynomial import Polynomial

def map_edt(p, edt, rs):
    rstep = np.mean(np.diff(rs))
    npos = p.shape[0]
    for i in range(npos):
        r,t = p[i,:].astype(int)
        # transformation from distance from the center
        p[i,0] = (edt[t,:,:].max() - rs[r]) / rstep
    return p

path_ext = '/media/guillermo/Expansion/Thesis GY/3. Analyzed files'
folder_masks = 'contour_masks'
folder_results = 'results'
folder_fluo = 'fluo'
folder_graphs = 'graphs'
folder_velocity = 'velocity_data'
vectors = ['pAAA', 'pLPT20&41', 'pLPT107&41', 'pLPT119&41']
channels = {'pAAA': {'rfp':0,'yfp':1,'cfp':2,'phase':3},
            'pLPT20&41': {'yfp':0,'cfp':1,'phase':2}, 
            'pLPT119&41': {'yfp':0,'cfp':1,'phase':2},
            'pLPT107&41': {'rfp':0,'yfp':1,'cfp':2,'phase':3}}
scopes = {'Tweez scope': 'TiTweez', 'Ti scope': 'Ti'}
dnas = {'pLPT20&pLPT41': 'pLPT20&41', 'pLPT119&pLPT41': 'pLPT119&41', 'pAAA': 'pAAA', 'pLPT107&pLPT41': 'pLPT107&41'}

exp_sum = pd.read_excel('../Notebooks/Exps_summary.xlsx')
exp_sum['formatted_dates'] = exp_sum['Date'].dt.strftime('%Y_%m_%d')
positions = pd.read_excel('../Notebooks/Positions.xlsx')

for i in exp_sum.index.values:
    exp_date = exp_sum.loc[i,'formatted_dates']
    vector = exp_sum.loc[i,'DNA']
    scope_name = exp_sum.loc[i,'Machine']
    df_pos = positions[(positions.Date == exp_sum.loc[i, 'Date']) & 
        (positions.DNA == vector) & 
        (positions.Machine == scope_name) & 
        (positions.Quality == 'Very good')]
    poss = df_pos.Position.unique()
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
    
    for pos in poss:
        print(f"Pos {pos}")
        print(f"{exp_date}_{scopes[scope_name]}_{vector}")
        path_scope = os.path.join(path_ext, scope_name)
        path = os.path.join(path_scope, exp_date)
        path_results = os.path.join(path, folder_results, f"pos{pos}")

        kymo = np.load(os.path.join(path_results,'kymo.npy'))
        dlrho = np.load(os.path.join(path_results,'dlrho.npy'))
        edt = np.load(os.path.join(path_results,'edt.npy'))
        nr = 64
        rw = 16
        rs = np.linspace(rw, edt.max(), nr)
        
        wdlkymo_rho = np.zeros_like(dlrho)
        nt, _, _ = dlrho.shape
        for c in range(fluo_chns):
            wdlkymo_rho[:,:,c] = warp(dlrho[:,:,c], map_edt, {'edt':edt, 'rs':rs})
        wdlkymo_rho[np.isnan(dlrho)] = np.nan
        np.save(os.path.join(path_results,'wdlkymo_rho.npy'), wdlkymo_rho)

        wkymo = np.zeros_like(kymo)
        nt, _, _ = kymo.shape
        for c in range(fluo_chns):
            wkymo[:,:,c] = warp(kymo[:,:,c], map_edt, {'edt':edt, 'rs':rs})
        wkymo[np.isnan(kymo)] = np.nan
        np.save(os.path.join(path_results,'wkymo.npy'), wkymo)