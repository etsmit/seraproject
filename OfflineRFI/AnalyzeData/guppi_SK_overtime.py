#--------------------------------------------------
#btlraw_SK.py
#Opens one output npy file from btl_to_raw.npy and performs SK 
#Takes four inputs:
#1: input filename (BL raw format) from my_dir
#2: npy file to save to
#--------------------------------------------------
# This does SK over a given number of spectra,
# so you can see SK over the course of the block
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
	sk_npy = my_dir + sys.argv[2]
else:
	sk_npy = sys.argv[2]



FFTLEN = float(sys.argv[3])
SK_ints = float(sys.argv[4])





#--------------------------------------
# Fun
#--------------------------------------


print('Opening file: '+inputFileName)
data= np.load(inputFileName)
print('Data shape: '+str(data.shape))
print('#--------------------------------------')


mismatch = data.shape[1] % FFTLEN
if mismatch != 0:
	print('Warning: FFTLEN does not divide the amount of time samples')
	print(str(mismatch)+' time samples at the end will be dropped')
kept_samples = data.shape[1] - mismatch

n=1

ints = np.float64(data.shape[1]/FFTLEN)
print('With given FFTLEN, there are '+str(ints)+' spectra per polarization per coarse channel')

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



print('Performing FFT...')

tot_sk_results=[]

for i in range(32):
	for j in range(2):
		print('Coarse Channel '+str(i))
		print('Polarization '+str(j))
		data_to_FFT = data[i,:kept_samples,j]
		data_to_FFT = data_to_FFT.reshape((FFTLEN,-1))
		s = np.abs(np.fft.fft(data_to_FFT,axis=0))**2
		mid_sk_results=[]
		for k in range(int(SK_timebins)):
			sk_spect = SK_EST(s[:,k*SK_ints:(k+1)*SK_ints],n,SK_ints)
			mid_sk_results.append(sk_spect)
		tot_sk_results.append(np.array(mid_sk_results))

tot_sk_results = np.array(tot_sk_results)
print('SK results shape: '+str(tot_sk_results.shape))


np.save(sk_npy, tot_sk_results)
print('SK spectra saved in '+str(sk_npy))
print('Data order(not including time axis): (Coarse) chan0,pol0 ; chan0,pol1 ; chan1,pol0 , etc...') 

print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))


print('Done!')












