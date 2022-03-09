import os
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, signal

from lmfit import models

#https://chrisostrouchov.com/post/peak_fit_xrd_python/

def update_spec_from_peaks(spec, model_indicies, distance, height, **kwargs):
    x = spec['x']
    y = spec['y']
    x_range = np.max(x) - np.min(x)
    peak_indicies,_ = signal.find_peaks(y, distance=distance, height=height)
    np.random.shuffle(peak_indicies)
    for peak_indicie, model_indicie in zip(peak_indicies.tolist(), model_indicies):
        model = spec['model'][model_indicie]
        if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
            params = {
                'height': y[peak_indicie],
                'sigma': x_range / len(x),
                'center': x[peak_indicie]
            }
            if 'params' in model:
                model.update(params)
            else:
                model['params'] = params
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
    return peak_indicies

def generate_model(spec):

    composite_model = None
    params = None
    x = spec['x']
    y = spec['y']
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    y_max = np.max(y)
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']: # for now VoigtModel has gamma constrained to sigma
            model.set_param_hint('sigma', min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1*y_max)
            model.set_param_hint('amplitude', min=1e-6)
            # default guess is horrible!! do not use guess()
            default_params = {
                prefix+'center': x_min + x_range * random.random(),
                prefix+'height': y_max * random.random(),
                prefix+'sigma': x_range * random.random()
            }
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func:  # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(**default_params, **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params

dat = np.loadtxt("/Users/alexlaroche/Desktop/PHYS 359/X-ray/PHYS359_xray/Cu_03-09-20.UXD")

spec = {
    'x': dat[:,0],
    'y': dat[:,1],
    'model': [
        {'type': 'VoigtModel'},
        {'type': 'VoigtModel'},
        {'type': 'VoigtModel'},
        {'type': 'VoigtModel'},
        {'type': 'VoigtModel'},
        {'type': 'VoigtModel'}
    ]
}

peaks_found = update_spec_from_peaks(spec, [0, 1, 2, 3, 4, 5], distance=20, height=20,sharex=True)
errs = np.sqrt(dat[:,1])
mask = errs != 0

max_iter = 100
thresh = 3.5
for i in range(max_iter):
    model, params = generate_model(spec)
    output = model.fit(spec['y'], params, x=spec['x'])
    chi2 = np.sum((dat[:,1][mask]-output.best_fit[mask])**2/errs[mask])/(len(dat[:,0][mask])-len(params))
    if chi2 <= thresh:
        print('chi2 =',np.round(chi2,2),'accepted')
        break
    elif chi2 > thresh:
        print('chi2 =',np.round(chi2,2),'rejected')
        if i==max_iter-1:
            print('Did not find a model satisfying chi2 <=',thresh)

fig, ax = plt.subplots(nrows=2, ncols = 1, gridspec_kw={'height_ratios': [3,1]})
plt.subplots_adjust(hspace=0)

ax[0].plot(dat[:,0],output.best_fit,color='r',label=np.round(chi2,2))
ax[0].plot(dat[:,0],dat[:,1],label='data')
ax[0].set_ylabel("Intensity [counts]")
ax[0].legend()

ax[1].errorbar(dat[:,0], dat[:,1]-output.best_fit, yerr = errs, ls = " ", marker = ".", capsize=2)
ax[1].set_xlabel(r"$2\theta$ [deg]")
ax[1].set_ylabel("Residuals")
ax[1].hlines(xmin = dat[:,0][0], xmax = dat[:,0][-1], y = 0, color = "k")

ax[0].scatter(dat[:,0][peaks_found],dat[:,1][peaks_found],marker='*')

ax[1].set_xlim(dat[:,0][0],dat[:,0][-1])
plt.savefig('full_fit.png')




