import os
import json
import numpy as np
import pandas as pd
from skimage.filters import gaussian
from skimage.io import imread, imsave
from skimage.transform import warp
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def crop_image(im_all, edt, nx, ny, pad):
    y,x = np.meshgrid(np.arange(nx), np.arange(ny))
    edt0 = edt[-1,:,:]
    minx = x[edt0>0].min()
    maxx = x[edt0>0].max()
    miny = y[edt0>0].min()
    maxy = y[edt0>0].max()
    w = maxx - minx
    w = int(w//2) * 2
    h = maxy - miny
    h = int(h//2) * 2

    #print(w,h)
    #pad = 32

    #crop_im_all = np.zeros((nt,w+2*pad-1,h+2*pad-1,nc))
    #crop_edt = np.zeros((nt,w+2*pad-1,h+2*pad-1))
    crop_im_all = np.zeros((nt,w+2*pad,h+2*pad,nc))
    crop_edt = np.zeros((nt,w+2*pad,h+2*pad))
    #print(crop_im_all.shape)
    #print(crop_edt.shape)

    for t in range(nt):
        tedt = edt[t,:,:]
        cx = int(x[tedt>0].mean())
        cy = int(y[tedt>0].mean())
        cim = im_all[t,cx - w//2 - pad:cx + w//2 + pad,cy - h//2 - pad:cy + h//2 + pad,:]
        crop_im_all[t,:,:,:] = cim
        crop_edt[t,:,:] = tedt[cx - w//2 - pad:cx + w//2 + pad,cy - h//2 - pad:cy + h//2 + pad]

    return crop_im_all, crop_edt

def map_func(p, edt, rs):
    rstep = np.mean(np.diff(rs))
    npos = p.shape[0]
    for i in range(npos):
        r,t = p[i,:].astype(int)
        # transformation from distance from the center
        p[i,0] = (edt[t,:,:].max() - rs[r]) / rstep
    return p

def get_kymo(im_all, edt, nr, rw):
    nt,nx,ny,nc = im_all.shape
    # dont construct the kymo from the edge, so it starts from rw to edt.max
    # the edge is not that reliable because of the mask, niose numbers at the edge
    rs = np.linspace(rw, edt.max(), nr)
    kymo = np.zeros((nt,nr,nc)) + np.nan
    #nkymo = np.zeros((nt,nr,3)) + np.nan
    for t in range(nt):
        for c in range(nc):
            for ri in range(nr):
                tedt = edt[t,:,:]
                idx = np.abs(tedt - rs[ri]) < rw
                if np.sum(idx)>0:
                    ntcim = im_all[t,:,:,c]
                    kymo[t,ri,c] = np.nanmean(ntcim[idx])
                    #ntcnim = normed_im[t,:,:,c]
                    #nkymo[t,ri,c] = np.nanmean(ntcnim[idx])
    return kymo

def get_dlkymo(kymo, nr, fluo_chns):    
    if fluo_chns == 3:
        kymo_rho = np.stack([kymo[:,:,0] / kymo[:,:,2], kymo[:,:,1] / kymo[:,:,2]], axis=2)
        cs = 2
    else:
        kymo_rho = kymo[:,:,0] / kymo[:,:,1]
        cs = 1

    lkymo_rho = np.log(kymo_rho)
    dlkymo_rho = np.zeros_like(lkymo_rho) + np.nan

    if fluo_chns == 3:
        for r in range(nr):
            for c in range(cs):
                idx = ~np.isnan(lkymo_rho[:,r,c])
                dlkymo_rho[idx,r,c] = savgol_filter(lkymo_rho[idx,r,c], 21, 3, deriv=1, axis=0)        
    else:
        for r in range(nr):
            idx = ~np.isnan(lkymo_rho[:,r])
            dlkymo_rho[idx,r] = savgol_filter(lkymo_rho[idx,r], 21, 3, deriv=1, axis=0)
    
    return dlkymo_rho

def plot_fluo_ratio(dlkymo_rho, path, rs, pos, fluo_chns):
    path_save = os.path.join(path, 'graphs', f'wdlkymo_rho_pos{pos}.png')
    wdlkymo_rho = np.zeros_like(dlkymo_rho)

    if fluo_chns == 3:
        for c in range(fluo_chns-1):
            wdlkymo_rho[:,:,c] = warp(dlkymo_rho[:,:,c], map_func, {'edt':edt, 'rs':rs})
        wdlkymo_rho[np.isnan(dlkymo_rho)] = np.nan
        
        plt.figure(figsize=(7,2))
        plt.subplot(1, 2, 1)
        plt.imshow(np.hstack([wdlkymo_rho[50:,::-1,0],wdlkymo_rho[50:,:,0]]).transpose(), 
                aspect='auto', 
                extent=[50,215,-edt.max(),edt.max()],
                vmin=-0.1, vmax=0.1,
                cmap='jet')
        plt.colorbar()
        plt.xlabel('Time (frames)')
        plt.ylabel('Radial distance (pixels)')
        plt.title('$d\mathrm{log}\\rho_{rc}/dt$')
        
        plt.subplot(1, 2, 2)
        plt.imshow(np.hstack([wdlkymo_rho[50:,::-1,1],wdlkymo_rho[50:,:,1]]).transpose(), 
                aspect='auto', 
                extent=[50,215,-edt.max(),edt.max()],
                vmin=-0.1, vmax=0.1,
                cmap='jet')
        plt.colorbar()
        plt.xlabel('Time (frames)')
        plt.ylabel('Radial distance (pixels)')
        plt.title('$d\mathrm{log}\\rho_{yc}/dt$')
        
        plt.tight_layout()
        plt.savefig(path_save, dpi=300)

    else:
        wdlkymo_rho[:,:] = warp(dlkymo_rho[:,:], map_func, {'edt':edt, 'rs':rs})
        wdlkymo_rho[np.isnan(dlkymo_rho)] = np.nan
        
        plt.figure(figsize=(7,2))
        #plt.subplot(1, 2, 1)
        plt.imshow(np.hstack([wdlkymo_rho[50:,::-1],wdlkymo_rho[50:,:]]).transpose(), 
                aspect='auto', 
                extent=[50,215,-edt.max(),edt.max()],
                vmin=-0.1, vmax=0.1,
                cmap='jet')
        plt.colorbar()
        plt.xlabel('Time (frames)')
        plt.ylabel('Radial distance (pixels)')
        plt.title('$d\mathrm{log}\\rho_{yc}/dt$')
        plt.tight_layout()
        plt.savefig(path_save, dpi=300)
    return wdlkymo_rho

#path_ext = '/media/guillermo/Expansion/Thesis GY/3. Analyzed files'
path_ext = '/media/c1046372/Expansion/Thesis GY/3. Analyzed files'

#scope_name = 'Tweez scope'
#scope_name = 'Ti scope'
#path_scope = os.path.join(path_ext, scope_name)
#exp_date = '2023_12_04'
#df_date = '2023-12-04'
#path = os.path.join(path_scope, exp_date)
#poss = [0,6,7,8,9,10,13,14,16,19,20,21,22,23,25,26,27,29,31,33,34]
#vectors = ['pAAA', 'pLPT20&41', 'pLPT107&41', 'pLPT119&41']
#vector = 'pLPT107&41'
folder_masks = 'contour_masks'
folder_results = 'results'
folder_fluo = 'fluo'
folder_graphs = 'graphs'
folder_velocity = 'velocity_data'

channels = {'pAAA': {'rfp':0,'yfp':1,'cfp':2,'phase':3},
            'pLPT20&41': {'yfp':0,'cfp':1,'phase':2}, 
            'pLPT119&41': {'yfp':0,'cfp':1,'phase':2},
            'pLPT107&41': {'rfp':0,'yfp':1,'cfp':2,'phase':3}}
# for file name
scopes = {'Tweez scope': 'TiTweez', 'Ti scope': 'Ti'}
dnas = {'pLPT20&pLPT41': 'pLPT20&41', 'pLPT119&pLPT41': 'pLPT119&41', 'pAAA': 'pAAA', 'pLPT107&pLPT41': 'pLPT107&41'}

# experiments metadata
with open('metadata.json') as f:
    metadata = json.load(f)

exp_sum = pd.read_excel('../Notebooks/Exps_summary.xlsx')
exp_sum['formatted_dates'] = exp_sum['Date'].dt.strftime('%Y_%m_%d')
positions = pd.read_excel('../Notebooks/Positions.xlsx')

for i in range(len(exp_sum)):
    exp_date = exp_sum.loc[i,'formatted_dates']
    vector = exp_sum.loc[i,'DNA']
    if vector == 'pAAA':
        continue
    else:
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

        for pos in poss:
            print(f"{exp_date}_{scopes[scope_name]}_{vector}")
            print(f"Pos {pos}")
            fname = f'{exp_date}_10x_1.0x_{dnas[vector]}_{scopes[scope_name]}_Pos{pos}.ome.tif'
            path_scope = os.path.join(path_ext, scope_name)
            path = os.path.join(path_scope, exp_date)
            path_im = os.path.join(path, fname)
            path_results = os.path.join(path, folder_results, f"pos{pos}")
            path_graphs = os.path.join(path, folder_graphs)

            im_all = imread(path_im)
            nt,nx,ny,nc = im_all.shape
            print(im_all.shape)

            # BG correction
            bg = np.zeros((nc,))
            for c in range(nc):
                bg[c] = im_all[0,:100,:100,c].mean()

            edt = np.load(os.path.join(path_results,'edt.npy'))
            edt = edt[:,:,:]

            pad = 32
            nr = 64
            rw = 16
            rs = np.linspace(rw, edt.max(), nr)
            
            #print("Crop image")
            #crop_im_all, crop_edt = crop_image(im_all, edt, nx, ny, pad)
            #im_all = crop_im_all
            #edt = crop_edt
            print("Kymo")
            kymo = get_kymo(im_all, edt, nr, rw)
            print("dlkymo_rho")
            #kymo = np.load(os.path.join(path_results, 'kymo.npy'))
            dlkymo_rho = get_dlkymo(kymo, nr, fluo_chns)
            print("wdlkymo_rho")
            wdlkymo_rho = plot_fluo_ratio(dlkymo_rho, path, rs, pos, fluo_chns)
            
            print("Saving files")
            ## save
            #np.save(os.path.join(path_results, 'crop_im_all.npy'), crop_im_all)
            #np.save(os.path.join(path_results, 'crop_edt.npy'), crop_edt)
            
            np.save(os.path.join(path_results, 'kymo.npy'), kymo)
            np.save(os.path.join(path_results, 'dlkymo_rho.npy'), dlkymo_rho)
            np.save(os.path.join(path_results, 'wdlkymo_rho.npy'), wdlkymo_rho)
            print("Files printed")
