##-----------------------------------------------
#Abell370_SK.py
#
#Performs SK on regrouped Abell370 data
#Inputs:
# 1: infile - npy file to open from rawvegas_to_npy.py
# 2: sk_results- npy file for saved SK spectra 
#-----------------------------------------------


import numpy as np
import os,sys
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize
import scipy.special

infile = sys.argv[1]
sk_npy = sys.argv[2]

#-----------------------------------------------
#functions


def SK_EST(a,n,m):
  #a = 2D power spectra - shape = (bandwidth,ints)
  #n should be 1
  #m = number of ints to use (from beginning of a)
  nchans= a.shape[0]
  d = 1#shape parameter (expect 1)
  sum1= np.sum(a[:,:m],axis=1)
  a2 = a**2
  sum2= np.sum(a2[:,:m],axis=1)
  #SK estimator spectrum
  sk_est = ((m*n*d+1)/(m-1))*(((m*sum2)/(sum1**2))-1)
  return sk_est

def upperRoot(x, moment_2, moment_3, p):
    upper = np.abs( (1 - sp.special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2)))-p)
    return upper

def lowerRoot(x, moment_2, moment_3, p):
    lower = np.abs(sp.special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2))-p)
    return lower

def SK_thresholds(M, N = 1, d = 1, p = 0.0013499):
    Nd = N * d
    #Statistical moments
    moment_1 = 1
    moment_2 = ( 2*(M**2) * Nd * (1 + Nd) ) / ( (M - 1) * (6 + 5*M*Nd + (M**2)*(Nd**2)) )
    moment_3 = ( 8*(M**3)*Nd * (1 + Nd) * (-2 + Nd * (-5 + M * (4+Nd))) ) / ( ((M-1)**2) * (2+M*Nd) *(3+M*Nd)*(4+M*Nd)*(5+M*Nd))
    moment_4 = ( 12*(M**4)*Nd*(1+Nd)*(24+Nd*(48+84*Nd+M*(-32+Nd*(-245-93*Nd+M*(125+Nd*(68+M+(3+M)*Nd)))))) ) / ( ((M-1)**3)*(2+M*Nd)*(3+M*Nd)*(4+M*Nd)*(5+M*Nd)*(6+M*Nd)*(7+M*Nd) )
    #Pearson Type III Parameters
    delta = moment_1 - ( (2*(moment_2**2))/moment_3 )
    beta = 4 * ( (moment_2**3)/(moment_3**2) )
    alpha = moment_3 / (2 * moment_2)

    error_4 = np.abs( (100 * 3 * beta * (2+beta) * (alpha**4)) / (moment_4 - 1) )
    x = [1]
    upperThreshold = sp.optimize.newton(upperRoot, x[0], args = (moment_2, moment_3, p))
    lowerThreshold = sp.optimize.newton(lowerRoot, x[0], args = (moment_2, moment_3, p))
    return lowerThreshold, upperThreshold


#-----------------------------------------------
  
  

#-----------------------------------------------
#Science!

print('Loading '+str(infile))
data = np.load(infile)
print('Loaded')
print('Data shape: '+str(data.shape))

ints = data.shape[1]
print(str(ints)+' integrations') 



#calculate thresholds
lt, ut = SK_thresholds(ints, N = 1, d = 1, p = 0.0013499)
print('Thresholds calculated')
print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))

sk_results=[]

#assuming 4 calibration scans of 61 ints each:
#lets just try for the first scan only

for i in range(16):
  print('Performing SK - round '+str(i+1)+' of 16')
  data_arr = np.transpose(data[i,245:545,:])
  sk_results.append(SK_EST(data_arr,1,ints))



np.save(sk_npy, sk_results)
print('SK spectra saved in '+str(sk_npy))


plt.plot(sk_results[0],'b+')
plt.title('SK on if1_pl1_cd0 scan 9')
plt.plot(np.zeros(len(sk_results[0]))+ut, 'r-')
plt.plot(np.zeros(len(sk_results[0]))+lt, 'r-')
plt.plot(np.zeros(len(sk_result[0]))+1, 'b-')


print('Done!')


  







