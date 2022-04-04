from Pb_Sn import *

#A

# A = np.loadtxt("data/A_10-03-22.UXD")

# bds = [ [40.5,43],
#         [47.5,50],
#         [68,74],
#         [84,88],
#         [89,92.5],
#         ]


# pars = [ [100,100,41.7,0.1,0.1,1,10000],
#          [150,150,48.6,0.1,0.1,1,10000],
#          [80,80,71,0.8,0.8,1,10000],
#          [80,80,85.7,0.1,0.1,1, 20,20,86.1,0.1,0.1,1, 100000],
#          [50,50,90.7,0.1,0.1,1,10000],
#         ]


# fit_peaks(A, pars, bds, "figures/Cu3Au-A/", "fits/Cu3Au-A_")



#B
B = np.loadtxt("data/B_10-03-22.UXD")

bds = [ [22,25],
        [32,35.5],
        [40,43.5],
        [46,51],
        [53,56.5],
        [59,62],
        [70,72],
        [75,77.5],
        [80,82.5],
        [85,87.5],
        [89.5,92],
        ]


pars = [ [100,100,23.7,0.1,0.1,1,10000],
         [150,150,33.8,0.1,0.1,1,10000],
         [400,400,41.7,0.1,0.1,1,10000],
         [100,100,48.6,0.1,0.1,1,100000],
         [60,60,54.7,0.1,0.1,1,10000],
         [30,30,60.5,0.1,0.1,1,10000],
         [100,100,71.1,0.1,0.1,1,10000],
         [20,20,76.1,0.1,0.1,1,10000],
         [10,10,81.1,0.1,0.1,1,10000],
         [100,100,85.9,0.1,0.1,1, 40,40,86.2,0.1,0.1,1, 10000],
         [30,30,90.8,0.1,0.1,1, 10,10,91,0.1,0.1,1, 10000],
        ]


fit_peaks(B, pars, bds, "figures/Cu3Au-B/", "fits/Cu3Au-B_")
