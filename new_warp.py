from skimage.transform import warp, EuclideanTransform, downscale_local_mean, rescale
import numpy as np
from skimage.io import imread, imsave
from scipy.ndimage import distance_transform_edt, laplace
from scipy.optimize import least_squares, fmin, minimize, leastsq
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os


def map_func(p, gx, gy, r, r0, mu0, sx, sy):
    R = r.max()
    gmag = np.sqrt(gx*gx + gy*gy)
    if r0==np.inf:
        vmag = mu0 * (R - r) / R
    elif r0>0:
        #vmag = mu0 * (np.exp(-r/r0) - np.exp(-R/r0)) / (1 - np.exp(-R/r0))
        vmag = mu0 * r0 / (R - r) * ((R - r - r0)*np.exp(-r/r0) + r0*np.exp(-R/r0))
        vmag[r==R] = 0
    else:
        vmag = 0
    #delr = laplace(r)
    #mu0 = dAdt / np.sum(np.exp(-r[r>0]/r0))
    #vmag = mu0 * r0 / (1 - r0 * delr) * np.exp(-r/r0)
    p[:,0] += gx * vmag / gmag + sx
    p[:,1] += gy * vmag / gmag + sy
    return p

def register(ref_mask, mask, im):
    ref_cx = px[ref_mask>0].mean()
    ref_cy = py[ref_mask>0].mean()
    cx = px[mask>0].mean()
    cy = py[mask>0].mean()
    shift = [ref_cx - cx, ref_cy - cy]
    tform = EuclideanTransform(translation=shift)
    regim = warp(im, tform.inverse, preserve_range=True)
    regmask = warp(mask, tform.inverse, preserve_range=True)>0
    return regim,regmask


def transform_image(im, edt, r0, mu0, sx, sy):
    mask = edt>0
    nmask = edt==0
    nedt = distance_transform_edt(nmask)
    redt = edt - nedt

    gx,gy = np.gradient(redt)
    args = {
        'gx':gx.ravel(), 
        'gy':gy.ravel(), 
        'r0':r0, 
        'r':redt.ravel(), 
        'mu0':mu0,
        'sx':sx,
        'sy':sy
    }
    wim = warp(im, map_func, args, preserve_range=True)
    return wim

def compare(im, t, r0, sx, sy, mu0):
    refmask = edt[tmin,:,:]>0
    im0 = im[t,:,:]
    mask0 = edt[t,:,:]>0
    im1 = im[t+1,:,:]
    mask1 = edt[t+1,:,:]>0
    regim0,regmask0 = register(refmask, mask0, im0)
    regim1,regmask1 = register(refmask, mask1, im1)
    regedt0 = distance_transform_edt(regmask0)
    #wim0 = transform_image(regim0, regedt0, r0,  mu0, sx, sy)
    wim0 = transform_image(im0, edt[t,:,:], r0,  mu0, sx, sy)
    #return wim0,regim0,regim1,regmask1
    return wim0,im0,im1,mask1


def fit_lsq(t, x0):
    #sxy = np.zeros((nt,2))
    #for t in range(tmin, tmax):
    #    # Refine registration
    #    def resid2(x):
    #        sx,sy = x
    #        wph0,regph0,regph1,regmask1 = compare(ph, t, np.inf, sx, sy)
    #        err = regph1[regmask1] - wph0[regmask1]
    #        return err.ravel()
    #    sx0 = 0,0
    #    sxopt,_,_,_,_ = leastsq(resid2, sx0, full_output=True)
    #    sx,sy  = sxopt
    #    sxy[t-tmin,:] = sxopt
    #    print(sx, sy)

    # Fit for r0
    def resid1(x):
        r0,sx,sy,mu0 = x
        r0 = np.exp(r0)
        #print(r0, sx, sy, mu0)
        residuals = np.array((0,))
        #for t in range(tmin, tmax):
        wph0,regph0,regph1,regmask1 = compare(ph, t, r0, sx, sy, mu0)
        err = regph1[regmask1] - wph0[regmask1]
        residuals = np.append(residuals, err.ravel())
        #print(f'RMSE = {np.sqrt(np.sum(residuals**2))}')
        return residuals
    popt,pcov,_,msg,_ = leastsq(resid1, x0, full_output=True) #least_squares(resid, x0)
    #print(msg)
    #r0 = res.x
    #print(res)
    #print(r0)
    #print(popt, pcov)
    pstd = np.sqrt(np.var(resid1(popt)) * pcov[0,0])
    return popt, pstd

def corr(t, r0, sx, sy, mu0):
    x = np.array((0,))
    y = np.array((0,))
    y0 = np.array((0,))
    #for t in range(tmin, tmax):
    wph0,regph0,regph1,regmask1 = compare(ph, t, r0, sx, sy, mu0)
    x = np.append(x, regph1[regmask1])
    y = np.append(y, wph0[regmask1])
    y0 = np.append(y0, regph0[regmask1])
    c = np.corrcoef(x,y)[0,1]
    #print(f'c={c}')
    c0 = np.corrcoef(x,y0)[0,1]
    #print(f'c0={c0}')
    return c,c0

path = '/home/campus.ncl.ac.uk/c1046372/Desktop/warp'


for pos in [0]:
    print(f"Pos {pos}")
    fname = f"2023_11_15_10x_1.0x_pLPT20&41_TiTweez_Pos{pos}.ome.tif"
    mname = f"mask_{fname}"
    path_file = os.path.join(path,fname)
    path_mask = os.path.join(path,'contour_masks',mname)

    im = imread(path_file)
    im = im.astype(float)
    nt,nx,ny,nc = im.shape   
    ph = im[:,:,:,2]
    fluo = im[:,:,:,:2]
    #nt,nx,ny,nc = fluo.shape
    #dtype = im.dtype

    path_results = os.path.join(path,'results',f"pos{pos}")
    area = np.load(os.path.join(path_results,'area.npy'))
    # radius = np.sqrt(area/np.pi)
    # dsarea = savgol_filter(area, 21, 3, deriv=1)
    # dsradius = savgol_filter(radius, 21, 3, deriv=1)

    edt = np.load(os.path.join(path_results,'edt.npy'))
    px,py = np.meshgrid(np.arange(nx), np.arange(ny))

    #idx = np.where((dsradius>1)*(radius>50))[0]
    #idx = np.where(radius>50)[0]
    #tmin,tmax = idx.min(),idx.max()
    
    tmin, tmax = 15, 100 #20, 21
    nt = tmax - tmin
    #print(f'tmin={tmin}, tmax={tmax}')


    r0 = np.zeros((nt,))
    c = np.zeros((nt,))
    c0 = np.zeros((nt,))
    std = np.zeros((nt,))
    mu0f = np.zeros((nt,))
    sxy = np.zeros((nt,2))

    tr0 = 128
    for t in range(tmin,tmax):
        print(f"Frame {t+1-tmin} / {tmax-tmin}")
        x0 = np.log(tr0),0,0,0.1 #dsradius[t]
        p, stdp = fit_lsq(t, x0)
        tr0,tsx,tsy,tmu0 = p
        stdp = np.sqrt((np.exp(stdp**2) - 1) * np.exp(2 * tr0 + stdp**2))
        tr0 = np.exp(tr0)
        mu0f[t-tmin] = tmu0
        sxy[t-tmin,0] = tsx
        sxy[t-tmin,1] = tsy
        r0[t-tmin] = tr0
        corr_corr, corr_no_fit = corr(t, tr0, tsx, tsy, tmu0)
        corr_corr_reg, corr_no_fit = corr(t, 0, tsx, tsy, tmu0)
        c[t-tmin] = corr_corr
        c0[t-tmin] = corr_corr_reg
        std[t-tmin] = stdp


    np.save(os.path.join(path_results,'r0_warp.npy'), r0)
    np.save(os.path.join(path_results,'r0_warp_std.npy'), std)
    np.save(os.path.join(path_results,'mu0.npy'), mu0f)
    np.save(os.path.join(path_results,'sxy.npy'), sxy)

    np.save(os.path.join(path_results,'c_warp.npy'), c)
    np.save(os.path.join(path_results,'c0_warp.npy'), c0)
    #print(f'c,c0 = {c}, {c0}')
