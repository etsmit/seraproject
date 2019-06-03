#--------------------------------------------------
#btlraw_SK.py
#Opens one output npy file from guppi_org_by_coarsechan and performs SK 
#Takes four inputs:
#1: input filename (GUPPI raw format) from my_dir
#2: filename base to save results
#--------------------------------------------------
# This does SK over a given number of spectra,
# so you can see SK over the course of all blocks
# time resolution is then m*sampling_rate*FFTLEN
#--------------------------------------------------

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
	result = my_dir + sys.argv[2]
else:
	result = sys.argv[2]

sk_result = result+'_SK.npy'
flags_result = result+'_flags.npy'



FFTLEN = float(sys.argv[3])
SK_ints = float(sys.argv[4])





#--------------------------------------
# Fun
#--------------------------------------


print('Opening file: '+inputFileName)
data= np.load(inputFileName)
print('Data shape: '+str(data.shape))
print('#--------------------------------------')


#Time sample - FFTLEN check
mismatch = data.shape[0] % FFTLEN
if mismatch != 0:
	print('Warning: FFTLEN does not divide the amount of frequency voltages')
	print(str(mismatch)+' data points at the end will be dropped')
kept_samples = data.shape[0] - mismatch

n=1

ints = np.float64(data.shape[0]/FFTLEN)
print('With given FFTLEN, there are '+str(ints)+' spectra per polarization')

# M to amount of spectra check
SK_ints_mismatch = ints % SK_ints
if SK_ints_mismatch != 0:
	print('Warning: SK_ints does not divide the number of ints')
	print(str(SK_ints_mismatch)+' spectra will be dropped')
kept_spectra = ints - SK_ints_mismatch

SK_timebins = kept_spectra/SK_ints
print('Leading to '+str(SK_timebins)+' SK time bins')

#calculate thresholds
print('Calculating SK thresholds...')
lt, ut = SK_thresholds(SK_ints, N = 1, d = 1, p = 0.0013499)
print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))


tot_sk_results=[]


for j in range(2):
	print('Polarization '+str(j))
	s = data[:kept_samples,j].reshape((FFTLEN,-1))
	s = np.abs(s)**2
	mid_sk_results=[]
	print('Performing SK...')
	for k in range(int(SK_timebins)):
		sk_spect = SK_EST(s[:,k*SK_ints:(k+1)*SK_ints],n,SK_ints)
		mid_sk_results.append(sk_spect)
	tot_sk_results.append(np.array(mid_sk_results))

tot_sk_results = np.array(tot_sk_results)
print('SK results shape: '+str(tot_sk_results.shape))

sk = tot_sk_results
np.save(sk_result, sk)
print('SK spectra saved in '+str(sk_result))

print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))

#------------------------------------------------
# APPLY FLAGS
# copied from SK_stats_time.py
#------------------------------------------------


flags = np.zeros(sk.shape)
tot_points = sk.size
flagged_pts = 0

#expecting a 3D sk results array for now (overtime)
pols = sk.shape[0]
SK_timebins = sk.shape[1]
finechans = sk.shape[2]

#look at every data point
for i in range(pols):
	for j in range(SK_timebins):
		for k in range(finechans):
			
			#is the datapoint outside the threshold?
			if (sk[i,j,k] < lt) or (sk[i,j,k] > ut):
				flagged_pts += 1
				flags[i,j,k] = 1

flagged_percent = (float(flagged_pts)/tot_points)*100
print(str(flagged_pts)+' datapoints were flagged out of '+str(tot_points))
print(str(flagged_percent)+'% of data outside acceptable ranges')

np.save(flags_result,flags)
print('Flags file saved to '+flags_result)


print('Done!')














