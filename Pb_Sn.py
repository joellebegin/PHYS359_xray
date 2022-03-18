import numpy as np
import matplotlib.pyplot as plt

from readData import voigt,fit_peak,fit_n_peaks

def get_bds(mus,widths):
    bds=[[0,0]]*len(mus)
    for i in range(len(bds)):
        bds[i] = [mus[i]-widths[i],mus[i]+widths[i]]
    return bds

'''
# FIT PB DATA (trivial since it's still FCC)

Pb_dat = np.loadtxt('data/Pb_10-03-22.UXD')

mus = [31.4,36.3,52.3,62.2,65.3,77.1,85.5,88.4,99.5,108]
widths = [0.8]*len(mus)
bds=get_bds(mus,widths)
amps = [100,100,10,10,2,2,2,2,2,2]

fit_n_peaks(Pb_dat,bds,mus,amps,'Pb')

# FIT SN DATA (bit more complicated because of double peaks)

Sn_dat = np.loadtxt('data/Sn_10-03-22.UXD')

mus=[30.7,32.1,44,45,55.4,62.65,63.9,64.65,72.45,73.25]
widths=[0.6,0.6,0.6,0.6,0.8,0.4,0.35,0.4,0.4,0.4]
bds=get_bds(mus,widths)
amps = [100,100,30,50,5,10,2,5,5,5]

fit_n_peaks(Sn_dat,bds,mus,amps,'Sn')'''

# FIT PB25 SN75 DATA (have to start doing multiple peak fits)
