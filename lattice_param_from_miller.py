import numpy as np
import matplotlib.pyplot as plt


#fucntion that takes one miller index (miller is of form [h,k,l]) d pair and spits out a with error
def a_computer(miller, two_theta, two_theta_err, lamb, lamb_err):
    #converting into theta
    theta = two_theta/2
    theta_err = two_theta_err/2
    
    #computing error from sin in lamb/(2sin(theta) = d
    denom = 2*np.sin(theta*np.pi/180)
    denom_err = 2*np.cos(theta*np.pi/180)*theta_err*np.pi/180
    
    #propagating error through division assuming zero covariance between params
    d = lamb/denom
    d_err = d*np.sqrt((lamb_err/lamb)**2 + (denom_err/denom)**2)
    
    #computing a (lattice parameter) using relation in lab manual
    miller_sum = np.sum([i**2 for i in miller])
    a = d*np.sqrt(miller_sum)
    a_err = d_err*(miller_sum)
    
    return([a, a_err])
    
    

#place to put in the 2theta values and errors
two_theta = np.array([2*21.71291062, 2*25.2745793 , 2*37.13089845, 2*45.03450581, 2*47.65147337])
two_theta_err = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

#the miller indices we got from assuming structure and seeing what works
miller_indices = [[1,1,1],[2,0,0],[2,2,0],[3,1,1],[2,2,2]]

lamb = 1.5418
lamb_err = 0.0005

a_vals = np.zeros((len(two_theta),2))


#looping over each data point and computing a from that
for i in range(len(two_theta)):
    a_temp = a_computer(miller_indices[i], two_theta[i], two_theta_err[i], lamb, lamb_err)
    a_vals[i,0] , a_vals[i,1] = a_temp

print(a_vals)
    
    
    
    
