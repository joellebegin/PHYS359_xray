import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.ion()

'''
WISHLIST: add a nice plotting function
'''


dat = np.loadtxt("Cu_03-09-20.UXD")
theta = dat[:,0]
cts = dat[:,1]

def lor(x,B,G,mu):
    lor = B*G/(2*np.pi*( (x - mu)**2 + (0.5*G)**2) )
    return lor

def gaus(x,A,mu,sig):
    gaus = A*(np.exp(-0.5*(x - mu)**2/sig**2))/(sig*np.sqrt(np.pi*2))
    return gaus

def gaus_and_lor(x,A,B,G, mu, sig, c):
    l = lor(x,B,G,mu)
    g = gaus(x,A,mu,sig)

    return l + g + c

def red_chi2(data, fit, errors, n_params):
    return np.sum((data - fit)**2/errors**2)/(len(data) - n_params)

if True:
    fig = plt.figure()
    plt.plot(theta, cts)
    plt.xlabel("2*theta [deg]")
    plt.ylabel("Intensity [counts]")
    plt.show()

def fit_and_plot(lim, p0, plot = True, plot_savepath=None):

    clip_dat = np.array([i for i in dat if (i[0]>lim[0] and i[0]<lim[1])]).T
    pars, cov = curve_fit(gaus_and_lor, clip_dat[0], clip_dat[1], p0=p0)

    errs = np.sqrt(clip_dat[1])
    fit = gaus_and_lor(clip_dat[0], *pars)
    res = clip_dat[1] - fit

    mask = errs != 0
    chi = red_chi2(clip_dat[1][mask], fit[mask], errs[mask], len(pars))

    if plot:
        thet_linspace = np.linspace(clip_dat[0][0], clip_dat[0][-1], 10000)
        fit_linsp = gaus_and_lor(thet_linspace, *pars)


        fig, ax = plt.subplots(nrows=2, ncols = 1, gridspec_kw={'height_ratios': [3,1]})
        plt.subplots_adjust(hspace=0)

        ax[0].errorbar(clip_dat[0], clip_dat[1], yerr = errs, color = "k", linestyle = "", marker = ".", label = str(chi))
        ax[0].plot(thet_linspace, fit_linsp)
        # ax[0].plot(thet2, gaus(thet2, pars[0], pars[3],pars[4]) + pars[5], label = "gaussian")
        # ax[0].plot(thet2, lor(thet2, pars[1], pars[2], pars[3]) + pars[5], label ="lor", color = "red")
        ax[0].set_ylabel("Intensity [counts]")
        ax[0].legend()

        ax[1].errorbar(clip_dat[0], res, yerr = errs, ls = " ", marker = ".")
        ax[1].set_xlabel("2*theta [deg]")
        ax[1].set_ylabel("Residuals")
        ax[1].hlines(xmin = lim[0], xmax = lim[1], y = 0, color = "k")
        
        if plot_savepath is not None:
            plt.savefig(plot_savepath, bbox_inches = "tight")
        else:
            plt.show()


#peak centers
p1 = 43.5
p2 = 50.5 
p3 = 74.2
p4 = 90
p5 = 95

pars = [
    [1000,1000,0.5,p1,0.5, 5],
    [500,500,0.5,p2,0.5, 5],
    [200,200,0.5,p3,0.5, 5],
    [100,100,0.5,p4,0.5, 5],
    [50,50,0.5,p5,0.5, 5]
]

lims = [
    [p1-3,p1+3],
    [p2-3,p2+3],
    [p3-3,p3+3],
    [p4-3,p4+3],
    [p5-3,p5+3]
]

for i in range(len(pars)):
    fit_and_plot(lims[i],pars[i], plot_savepath=f"peak_{i}.png")