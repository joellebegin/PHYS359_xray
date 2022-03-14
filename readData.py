import numpy as np
import matplotlib.pyplot as plt

from scipy.special import voigt_profile
from scipy.optimize import curve_fit

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
    # p0 = mu,alpha,gamma,C
    cut = dat[np.argmin(np.abs(dat[:,0]-bds[0])):np.argmin(np.abs(dat[:,0]-bds[1]))]
    popt,pcov = curve_fit(voigt,cut[:,0],cut[:,1],p0)
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
    return popt,pcov

dat = np.loadtxt("data/Cu_09-03-22.UXD")

bds = [[41,45],[48,52],[72,76],[88,92],[93,98]]
mus = [43.5,50.5,74.2,90,95]
amps = [100,10,10,10,2]

for i in range(len(bds)):
    fit_peak(dat,bds[i],[amps[i],mus[i],1,1,0.1],plot=True,filename=f"figures/peak_{i}_voight.png")