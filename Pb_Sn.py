import numpy as np
import matplotlib.pyplot as plt

from readData import *

def get_bds(mus,widths):
    bds=[[0,0]]*len(mus)
    for i in range(len(bds)):
        bds[i] = [mus[i]-widths[i],mus[i]+widths[i]]
    return bds


def two_voigt(x,A1,B1,x01,sigma1, gamma1,C1,
                A2,B2,x02,sigma2, gamma2,C2):
    '''
    makes a function w/ many voigt peaks

    x: data array

    n: int, number of peaks we will be fitting

    params: arr, params for each peak, must be params%n=0
    '''
    func = voigt(x,A1,B1,x01,sigma1, gamma1,C1) + voigt(x,A2,B2,x02,sigma2, gamma2,C2)

    return func


def three_voigt(x,A1,B1,x01,sigma1, gamma1,C1,
                  A2,B2,x02,sigma2, gamma2,C2,
                  A3,B3,x03,sigma3, gamma3,C3):
    '''
    makes a function w/ many voigt peaks

    x: data array

    n: int, number of peaks we will be fitting

    params: arr, params for each peak, must be params%n=0
    '''
    func = voigt(x,A1,B1,x01,sigma1, gamma1,C1) + voigt(x,A2,B2,x02,sigma2, gamma2,C2) + voigt(x,A3,B3,x03,sigma3, gamma3,C3)

    return func


def fit_multiple_peaks(dat,bds,p0,fit_func,plot=True,filename=None):
    # p0 = A,B,x0,sigma,gamma,C, maxfev
    
    cut = dat[np.argmin(np.abs(dat[:,0]-bds[0])):np.argmin(np.abs(dat[:,0]-bds[1]))]
    popt,pcov = curve_fit(fit_func,cut[:,0],cut[:,1],p0[:-1],maxfev=p0[-1])
    if plot==True:
        fig, ax = plt.subplots(nrows=2, ncols = 1, gridspec_kw={'height_ratios': [3,1]}, figsize=(10,5))
        plt.subplots_adjust(hspace=0)
        errs = np.sqrt(cut[:,1])
        ax[0].errorbar(cut[:,0],cut[:,1],yerr=errs,ls='')
        mask = errs != 0
        fit = fit_func(cut[:,0],*popt)
        chi2 = np.sum((cut[:,1][mask]-fit[mask])**2/errs[mask])/(len(cut[mask])-len(p0))
        ax[0].plot(cut[:,0],fit,color='r',label=np.round(chi2,2))
        ax[0].set_ylabel("Intensity [counts]")
        ax[0].legend()
        ax[0].legend(fontsize=15)
        ax[1].errorbar(cut[:,0],cut[:,1]-fit,yerr=errs,ls='')
        ax[1].axhline(0,color='k')
        ax[1].set_xlabel(r"$2\theta$ [deg]")
        ax[1].set_ylabel("Residuals")
        if filename is not None:
            plt.savefig(filename, bbox_inches = "tight")
            # plt.show()
        else:
            plt.show()

    return cut,popt,pcov


def fit_peaks(dat, pars, bds,fig_savepath,file_savepath):
    for i,p in enumerate(pars):
        if len(p)==7:
            cut,popt,pcov = fit_multiple_peaks(dat, fit_func = voigt, bds=bds[i],p0=pars[i],filename=fig_savepath+ f"peak_{i}.png")
            np.savetxt(file_savepath+f"mu_err_{i}.txt", np.array([popt[2],pcov[2,2]]))
        elif len(p) == 13:
            cut,popt,pcov = fit_multiple_peaks(dat, fit_func = two_voigt, bds=bds[i],p0=pars[i],filename=fig_savepath + f"peak_{i}.png")
            np.savetxt(file_savepath+f"mu_err_{i}.txt", np.array([[popt[2],pcov[2,2]],
                                                    [popt[8],pcov[8,8]]]) )
        elif len(p) == 19:
            cut,popt,pcov = fit_multiple_peaks(dat, fit_func = three_voigt, bds=bds[i],p0=pars[i],filename=fig_savepath + f"peak_{i}.png")
            np.savetxt(file_savepath+f"mu_err_{i}.txt", np.array([[popt[2],pcov[2,2]],
                                                    [popt[8],pcov[8,8]],
                                                    [popt[14],pcov[14,14]]]) )


sn=False

if sn:
    Sn_dat = np.loadtxt('data/Sn_10-03-22.UXD')

    pars = [
        [50,50,30.7,1,1,1, 50,50,32,1,1,1, 10000],
        [30,30,43.95,1,1,1, 40,40,44.98,1,1,1, 10000],
        [5,5,55.4,1,1,1, 10000],
        [5,5,62.6,1,1,1 ,1,1,63.8,1,1,1, 5,5,64.6,1,1,1, 10000],
        [10,10,72.4,1,1,1, 10,10,73.2,1,1,1, 10000]
    ]

    bds = [
        [28,34],
        [42,46],
        [54,57],
        [60,67],
        [71,75]
    ]

    fit_peaks(Sn_dat, pars, bds, "figures/Sn/", "fits/Sn_")



#===================== Pb =====================#

pb=False
if pb:
    Pb_dat = np.loadtxt("data/Pb_10-03-22.UXD")

    pars = [
           [100,100,31.3,1,1,1],
           [50,50,36.3,1,1,1],
           [10,10,52.3,0.8,0.8,1],
           [10,10,62.3,1,1,1],
           [3,3,65.3,1,1,1],
           [1,1,77.1,1,1,1]
    ]

    bds = [      
          [30,32.5],
          [35,37.5],
          [51,53.5],
          [61,63.5],
          [64,66],
          [76,78]
     ]

    fit_peaks(Pb_dat, pars, bds, "figures/Pb/", "fits/Pb_")


#===================== Pb25_Sn75 =====================#

pb25sn75=False

if pb25sn75:
    
    Pb25Sn75_dat = np.loadtxt('data/Pb25_Sn75_10-03-22.UXD')

    pars = [
        [40,40,30.6,3,3,3, 300,300,31.3,.6,.6,1, 100,100,32.05,.8,.8,1,100000], #bad!!
        [20,20,36.2,1,1,1,10000],
        [15,15,43.85,1,1,1, 50,50,44.9,1,1,1,100000],
        [50,50,52.3,1,1,1,10000],
        [10,10,55.38,1,1,1,10000],
        [5,5,62.36,1,1,1,10000], #could be better?
        [1,1,63.9,1,1,1, 10,10,64.68,1,1,1, 10,10,65.3,1,1,1,100000] #bad, may just dump?
    ]

    bds = [      
        [29,34],
        [35,38],
        [42,47],
        [50,54],
        [54,58],
        [61.3,63.4],
        [63.4,66]
    ]

    fit_peaks(Pb25Sn75_dat, pars, bds, "figures/Pb25Sn75/", "fits/Pb25Sn75_")


#===================== Pb50_Sn50 =====================#

pb50sn50=False

if pb50sn50:
    dat = np.loadtxt('data/Pb50_Sn50_10-03-22.UXD')


    pars = [
        [10,10,30.7,3,3,3, 300,300,31.4,1,1,1, 100,100,32.,1,1,1,10000], #bad!!!!
        [100,100,36.7,1,1,1,10000],
        [20,20,43.95,1,1,1, 50,50,44.9,1,1,1,10000], 
        [50,50,52.3,1,1,1, 10,10,55.4,1,1,1,10000],
        [10,10,62.4,1,1,1,10000], #could be better?
    ]

    bds = [      
        [30,33],
        [35,38],
        [42,47],
        [50,57],
        [61.3,63.4],
    ]

    fit_peaks(dat, pars, bds, "figures/Pb50Sn50/", "fits/Pb50Sn50_")


#===================== Pb50_Sn50 =====================#

pb75sn25=False

if pb75sn25:
    dat = np.loadtxt('data/Pb75_Sn25_10-03-22.UXD')


    pars = [
        # [100,100,31.4,1,1,1, 10,10,32.1,1,1,1, 100000], #bad!!!!
        # [100,100,36.4,1,1,1,10000],
        # [20,20,52.3,1,1,1,10000], 
        [100,100,62.3,1,1,1, 10,10,65.4,1,1,1,10000],
        # [10,10,62.4,1,1,1,10000], #could be better?
    ]

    bds = [      
        # [30,33],
        # [35.5,37.5],
        # [51,54],
        [60,67],
        # [61.3,63.4],
    ]

    fit_peaks(dat, pars, bds, "figures/Pb75Sn25/", "fits/Pb75Sn25_")
