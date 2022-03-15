import numpy as np
import matplotlib.pyplot as plt

from scipy.special import voigt_profile
from scipy.optimize import curve_fit
import pickle

def Gaussian(x, *p):
    A, x0, sigma, C = p
    return A*(np.sqrt(2*np.pi*sigma))*np.exp(-(x-x0)**2/(2*sigma**2))+C

def Lorentzian(x, *p):
    A, x0, gamma, C = p
    return A/np.pi*gamma/((x-x0)**2+gamma**2)+C

def voigt(x,A,x0,sigma, gamma,C):
    G = np.fft.fft(Gaussian(x,1,x0,sigma,C))
    L = np.fft.fft(Lorentzian(x,1,x0,gamma,C))
    V = A*np.real(np.fft.fftshift(np.fft.ifft(G*L)))
    return V

def fit_peak(dat,bds,p0,plot=True,filename=None):
    # p0 = A,x0,sigma,gamma,C
    cut = dat[np.argmin(np.abs(dat[:,0]-bds[0])):np.argmin(np.abs(dat[:,0]-bds[1]))]
    popt,pcov = curve_fit(voigt,cut[:,0],cut[:,1],p0,maxfev=10000)
    if plot==True:
        fig, ax = plt.subplots(nrows=2, ncols = 1, gridspec_kw={'height_ratios': [3,1]}, figsize=(10,5))
        plt.subplots_adjust(hspace=0)
        errs = np.sqrt(cut[:,1])
        ax[0].errorbar(cut[:,0],cut[:,1],yerr=errs,ls='')
        mask = errs != 0
        fit = voigt(cut[:,0],*popt)
        chi2 = np.sum((cut[:,1][mask]-fit)**2/errs[mask])/(len(cut[mask])-len(p0))
        ax[0].plot(cut[:,0],fit,color='r',label=np.round(chi2,2))
        ax[0].set_ylabel("Intensity [counts]")
        ax[0].legend()
        ax[0].legend(fontsize=15)
        ax[1].errorbar(cut[:,0],cut[:,1]-fit,yerr=errs,ls='')
        ax[1].axhline(0,color='k')
        ax[1].set_xlabel(r"$2\theta$ [deg]")
        ax[1].set_ylabel("Residuals")
        if filename is not None:
            plt.savefig(filename)
    return cut,popt,pcov

def fit_n_peaks(data,bds,mus,amps,material_name):

    popt=[0]*len(bds)
    pcov=[0]*len(bds)
    cut=[0]*len(bds)

    for i in range(len(bds)):
        path = 'figures/' + material_name + f'/peak_{i}.png'
        cut_i,popt_i,pcov_i = fit_peak(data,bds[i],[amps[i],mus[i],1,1,.1],plot=True,filename=path)
        cut[i]=cut_i
        popt[i]=popt_i
        pcov[i]=pcov_i

    dict = {'cut':cut,'popt':popt,'pcov':pcov}

    filename = 'fits/' + material_name + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
    # to load data:
    #with open('filename.pickle', 'rb') as handle:
    #    b = pickle.load(handle)

# FIT PURE CU DATA

Cu_dat = np.loadtxt("data/Cu_09-03-22.UXD")

bds = [[41,45],[48,52],[72,76],[88,92],[93,98]]
mus = [43.5,50.5,74.2,90,95]
amps = [100,10,10,10,2]

fit_n_peaks(Cu_dat,bds,mus,amps,'Cu')

# FIT PURE NI DATA

Ni_dat = np.loadtxt("data/Ni_09-03-22.UXD")

bds = [[41,48],[48,55],[74,79],[91,95],[97.5,99.5]]
mus = [44.5,52,76.5,93,98.5]
amps = [1000,10,10,10,5]

fit_n_peaks(Ni_dat,bds,mus,amps,'Ni')

# FIT CU25 NI75 DATA

Cu25_Ni75_dat = np.loadtxt("data/Cu25_Ni75_09-03-22.UXD")

bds = [[41,48],[48,55],[73,79],[90,94],[96,99.5]]
mus = [44,52,76,92.5,97.5]
amps = [1000,10,10,10,5]

fit_n_peaks(Cu25_Ni75_dat,bds,mus,amps,'Cu25_Ni75')

# FIT CU50 NI50 DATA

Cu50_Ni50_dat = np.loadtxt("data/Cu50_Ni50_10-03-22.UXD")

bds = [[41,47],[48,55],[73,79],[90,94],[95,99]]
mus = [44,51,75,92,97]
amps = [500,100,100,10,5]

fit_n_peaks(Cu50_Ni50_dat,bds,mus,amps,'Cu50_Ni50')

# FIT CU75 NI25 DATA

Cu75_Ni25_dat = np.loadtxt("data/Cu75_Ni25_10-03-22.UXD")

bds = [[41,46],[48,55],[73,77],[88.5,92.5],[94.5,98.5]]
mus = [43.5,51,75,90.5,96.5]
amps = [500,100,50,10,5]

fit_n_peaks(Cu75_Ni25_dat,bds,mus,amps,'Cu75_Ni25')