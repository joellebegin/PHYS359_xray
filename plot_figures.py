import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import pickle
from readData import voigt

# PEAK FIT PLOT

with open('fits/Cu.pickle', 'rb') as handle:
    Cu_fit = pickle.load(handle)

cut=Cu_fit['cut'][2]
popt=Cu_fit['popt'][2]
pcov=Cu_fit['pcov'][2]

fig, ax = plt.subplots(nrows=2, ncols = 1, gridspec_kw={'height_ratios': [3,1]}, figsize=(6,6),sharex=True)
plt.subplots_adjust(hspace=0)
errs = np.sqrt(cut[:,1])
ax[0].errorbar(cut[:,0],cut[:,1],yerr=errs,ls='',marker='.',capsize=3,label='Cu data')
mask = errs != 0
fit = voigt(cut[:,0],*popt)
chi2 = np.sum((cut[:,1][mask]-fit)**2/errs[mask])/(len(cut[mask])-len(popt))
label = r'$\chi_2=$'+str(np.round(chi2,2))
x=np.linspace(cut[:,0][0],cut[:,0][-1],1000)
ax[0].plot(x,voigt(x,*popt),color='r',label=label)
ax[0].set_ylabel("Intensity [counts]",fontsize=20)
ax[0].legend()
ax[0].legend(fontsize=15)
ax[1].errorbar(cut[:,0],cut[:,1]-fit,yerr=errs,ls='',capsize=3,marker='.')
ax[1].axhline(0,color='k')
ax[1].set_xlabel(r"$2\theta$ [deg]",fontsize=20)
ax[1].set_ylabel("Residuals",fontsize=20)
ax[0].tick_params(axis='both', which='major', labelsize=15)
ax[1].tick_params(axis='both', which='major', labelsize=15)
ax[0].set_xlim(x[0],x[-1])
plt.savefig('figures/interim_1/peak_fit.png',dpi=300,bbox_inches='tight')
plt.clf()

# LATTICE PARAM PLOT

materials = ['Ni','Cu25_Ni75','Cu50_Ni50','Cu75_Ni25','Cu']

fits=[0]*len(materials)
angles=[0]*len(materials)
errs=[0]*len(materials)

for i,material in enumerate(materials):
    
    filename = 'fits/' + material + '.pickle'
    with open(filename, 'rb') as handle:
        fits[i] = pickle.load(handle)
    
    angles[i] = np.array([fits[i]['popt'][j][2] for j in range(len(fits[i]['popt']))])/2
    errs[i] = np.array([fits[i]['pcov'][j][2][2] for j in range(len(fits[i]['pcov']))])/2
        
miller=np.sqrt(np.array([3,4,8,11,12]))

lam=1.5444256
err_lam=0.0000019

def linear(x,a,x0):
    return a*np.sin((x-x0)*np.pi/180)

color=['blue','green','purple','pink','red']
names=['Ni',r'$25\%$ Cu, $75\%$ Ni',r'$50\%$ Cu, $50\%$ Ni',r'$75\%$ Cu, $25\%$ Ni','Cu',]

fig, ax = plt.subplots(nrows=2, ncols = 1, figsize=(6,6), gridspec_kw={'height_ratios': [3,1]})
percent_Cu = np.array([0,25,50,75,100])
a=[0]*5
a_err=[0]*5

for i in range(len(fits)):
    popt,pcov=curve_fit(linear,angles[i],miller*lam/2,p0=[3.61,0],
                    sigma=err_lam*miller/2,absolute_sigma=True)
    a[i]=popt[0]
    a_err[i]=np.sqrt(pcov[0][0])
    
a=np.array(a)
a_err=np.array(a_err)

def vegard(x,A=a[0],B=a[-1]):
    return (1-x)*A + x*B
x_Cu=percent_Cu/100
popt,pcov=curve_fit(vegard,percent_Cu/100,a,p0=[a[0],a[-1]],
                    sigma=a_err,absolute_sigma=True)
plt.subplots_adjust(hspace=0)

ax[0].plot(percent_Cu,a,color='k',ls='--')
ax[1].plot(percent_Cu,a-vegard(x_Cu),color='k',ls='--')

for j in range(len(fits)):
        
    label = str(format(a[j], '.5f')) + ' (' + str(int(1e6*a_err[j])) + ') $\AA$'
    ax[0].errorbar(percent_Cu[j],a[j],label=label,color=color[j],yerr=a_err[j],
               ls='--',marker='.',markersize=10)
    ax[0].set_ylabel(r'$a$ [$\AA$]',fontsize=20)
    ax[1].set_xlabel(r'$\%$ of material Cu',fontsize=20)
    ax[0].legend(fontsize=12)
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    ax[1].errorbar(percent_Cu[j],a[j]-vegard(x_Cu[j]),label=label,color=color[j],yerr=a_err[j],
               ls='--',marker='.',markersize=10)
    ax[1].set_ylabel('Residuals',fontsize=20)
    ax[1].axhline(0,color='k')
    ax[1].tick_params(axis='both', which='major', labelsize=15)

ax[0].plot(x_Cu*100,vegard(x_Cu),'k')
plt.savefig('figures/interim_1/lattice_param.png',dpi=300,bbox_inches='tight')
plt.clf()

# DATA PLOT

fig, ax = plt.subplots(nrows=1, ncols = 1, figsize=(6,6))
data=np.loadtxt('data/Cu_09-03-22.UXD')
ax.errorbar(data[:,0],data[:,1],yerr=np.sqrt(data[:,1]),marker='.')
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlabel(r'$2\theta$ [deg]',fontsize=20)
ax.set_ylabel('X-ray photon counts',fontsize=20)
ax.set_xlim(40,100)

plt.savefig('figures/interim_1/data_miller.png',dpi=300,bbox_inches='tight')