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

from SK_in_Python import *


my_dir = '/home/scratch/esmith/RFI_MIT/'

#pulls from my scratch directory if full path not given
if sys.argv[1][0] != '/':
	inputFileName = my_dir + sys.argv[1]
else:
	inputFileName = sys.argv[1]

#same for  output destination
if sys.argv[2][0] != '/':
	sk_npy = my_dir + sys.argv[2]
else:
	sk_npy = sys.argv[2]

#-----------------------------------------------
#Science!

print('Loading '+str(inputFileName))
data = np.load(inputFileName)
print('Loaded')
print('Data shape: '+str(data.shape))

ints = np.float64(data.shape[1])
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
	print(data_arr.shape)
	sk_results.append(SK_EST(data_arr,1,ints))



np.save(sk_npy, sk_results)
print('SK spectra saved in '+str(sk_npy))


plt.plot(sk_results[0],'b+')
plt.title('SK on if1_pl1_cd0 scan 9')
plt.plot(np.zeros(len(sk_results[0]))+ut, 'r-')
plt.plot(np.zeros(len(sk_results[0]))+lt, 'r-')
plt.plot(np.zeros(len(sk_results[0]))+1, 'b-')
plt.show()


print('Done!')


  







