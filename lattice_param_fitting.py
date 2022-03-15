import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit

# FIT LATTICE PARAMETER FOR CU NI MATERIALS

materials = ['Ni','Cu25_Ni75','Cu50_Ni50','Cu75_Ni25','Cu']

fits=[0]*len(materials)
angles=[0]*len(materials)
errs=[0]*len(materials)

for i,material in enumerate(materials):
    filename = 'fits/' + material + '.pickle'
    with open(filename, 'rb') as handle:
        fits[i] = pickle.load(handle)
    
    angles[i] = np.array([fits[i]['popt'][j][1] for j in range(len(fits[i]['popt']))])/2
    errs[i] = np.array([fits[i]['pcov'][j][1][1] for j in range(len(fits[i]['pcov']))])

fig, ax = plt.subplots(nrows=2, ncols = 1, gridspec_kw={'height_ratios': [3,1]}, figsize=(10,5))
plt.subplots_adjust(hspace=0)
        
miller=np.sqrt(np.array([3,4,8,11,12]))

lam=1.5444256
err_lam=0.0000019

def linear(x,a,x0):
    return a*np.sin((x-x0)*np.pi/180)

color=['blue','green','purple','pink','red']
names=['Ni',r'$25\%$ Cu, $75\%$ Ni',r'$50\%$ Cu, $50\%$ Ni',r'$75\%$ Cu, $25\%$ Ni','Cu',]

for i in range(len(fits)):
    ax[0].errorbar(np.sin(angles[i]*np.pi/180),miller*lam/2,yerr=err_lam*miller/2,color=color[i],
               xerr=np.cos(angles[i]*np.pi/180)*np.pi/180*errs[i]/2,marker='.',markersize=4,ls='')
    popt,pcov=curve_fit(linear,angles[i],miller*lam/2,p0=[3.61,0],
                    sigma=err_lam*miller/2*5,absolute_sigma=True)
    x=np.linspace(15,55,1000)
    label = names[i] + r': $a=$ ' + str(np.round(popt[0],5)) + ' $\pm$ ' + str(np.round(np.sqrt(pcov[0][0]),6)) + ' $\AA$'
    ax[0].plot(np.sin(x*np.pi/180),linear(x,*popt),label=label,color=color[i])
    ax[1].errorbar(np.sin(angles[i]*np.pi/180),miller*lam/2-linear(angles[i],*popt),
               yerr=err_lam*miller/2,xerr=np.cos(angles[i]*np.pi/180)*np.pi/180*errs[i]/2,
               marker='.',markersize=4,ls='',color=color[i])
    ax[1].axhline(0,color='k')
    ax[1].set_ylabel('Residuals',fontsize=15)
    ax[0].set_ylabel(r'$\frac{\lambda}{2}\sqrt{h^2+k^2+l^2}$ [$\AA$]',fontsize=15)
    ax[1].set_xlabel(r'$\sin\theta$',fontsize=20)
    ax[0].legend()

plt.savefig('figures/lattice_param.png')


