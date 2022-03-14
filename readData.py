import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, signal

import math
import random

from lmfit import models

# plt.ion()

'''
WISHLIST: add a nice plotting function
'''


dat = np.loadtxt("Cu_03-09-20.UXD")

def peak_finder(spec, model_indicies, distance, height, **kwargs):
    '''
    Finds peaks in given data. 

    Params:
    -------
    spec: dictionary
        dictionary w/ data + fit models

    model_indices: list
        just a list indexing the models

    distance: int
        minimum distance between peaks. Prob dont need here since
        only doing one peak at a time

    height: height of peaks
    '''
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
    '''
    I have absolutely no idea what this function does, and don't feel like deciphering it
    to add good comments. Idk this whole framework seems kinda overkill-- was nice
    when we were trying to fit all peaks at once, but now we're doing one at a time.
    I think we could probably write a way simpler version???
    '''

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


def fit_one_peak(lims):
    '''
    Fits just one peak. 

    Params:
    -------
    lims: 2d array/list
        Gives window for peak

    Returns:
    --------
    '''
    #clipping data according to given limits
    clip_dat = np.array([i for i in dat if (i[0]>lims[0] and i[0]<lims[1])]).T

    spec = {
        'x': clip_dat[0],
        'y': clip_dat[1],
        'model': [
            {'type': 'VoigtModel'},
            {'type': 'VoigtModel'}
        ]
    }

    peaks_found = peak_finder(spec, [0, 1], distance=20, height=20,sharex=True)
    errs = np.sqrt(clip_dat[1])
    mask = errs != 0

    max_iter = 300
    thresh = 3.5
    for i in range(max_iter):
        model, params = generate_model(spec)
        output = model.fit(spec['y'], params, x=spec['x'])
        chi2 = np.sum((clip_dat[1][mask]-output.best_fit[mask])**2/errs[mask])/(len(clip_dat[0][mask])-len(params))
        if chi2 <= thresh:
            # print('chi2 =',np.round(chi2,2),'accepted')
            break
        elif chi2 > thresh:
            # print('chi2 =',np.round(chi2,2),'rejected')
            if i==max_iter-1:
                print('Did not find a model satisfying chi2 <=',thresh)

        
    fig, ax = plt.subplots(nrows=2, ncols = 1, gridspec_kw={'height_ratios': [3,1]})
    plt.subplots_adjust(hspace=0)

    ax[0].plot(clip_dat[0],output.best_fit,color='r',label=np.round(chi2,2))
    ax[0].plot(clip_dat[0],clip_dat[1],label='data')
    ax[0].set_ylabel("Intensity [counts]")
    ax[0].legend()

    ax[1].errorbar(clip_dat[0], clip_dat[1]-output.best_fit, yerr = errs, ls = " ", marker = ".", capsize=2)
    ax[1].set_xlabel(r"$2\theta$ [deg]")
    ax[1].set_ylabel("Residuals")
    ax[1].hlines(xmin = clip_dat[0][0], xmax = clip_dat[0][-1], y = 0, color = "k")

    ax[0].scatter(clip_dat[0][peaks_found],clip_dat[1][peaks_found],marker='*')
    plt.show()



#peak centers
p1 = 43.5
p2 = 50.5 
p3 = 74.2
p4 = 90
p5 = 95

win_len = 2
lims = [
    [p1-win_len,p1+win_len],
    [p2-win_len,p2+win_len],
    [p3-win_len,p3+win_len],
    [p4-win_len,p4+win_len],
    [p5-win_len,p5+win_len]
]

print(fit_one_peak(lims[1]))