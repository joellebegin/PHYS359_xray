# PHYS 359 X-ray Diffraction Code Base
### You will need to pip install [lmfit](https://lmfit.github.io/lmfit-py/) to run this code.

## XRD Fitting
XRD fitting is performed using code written by [Chris Ostrouchov](https://chrisostrouchov.com/post/peak_fit_xrd_python/).
We typically use n+1 functions to fit a dataset with n peaks. The additional function accounts for background. We typically make use of a Voigt function (a convolution of Lorentzian and Gaussian functions) to model individual XRD peaks. Best fit models are saved to 'outputs'. For documentation on the make up of output files, and how to manipulate them, check out the lmfit documentation (link above).
