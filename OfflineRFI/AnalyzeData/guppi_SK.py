#--------------------------------------------------
#btlraw_SK.py
#Opens one output npy file from btl_to_raw.npy and performs SK 
#Takes two inputs:
#1: input filename (BL raw format) from my_dir
#2: npy file to save to
#--------------------------------------------------
# This does SK on the entirety of the block in one go
# so no time resolution finer than the size of the block
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



FFTLEN = int(sys.argv[3])


#--------------------------------------
# Fun
#--------------------------------------


print('Opening file: '+inputFileName)
data= np.load(inputFileName)
print('Data shape: '+str(data.shape))
print('#--------------------------------------')


num_coarsechan = data.shape[0]
num_timesamples= data.shape[1]
num_pol = data.shape[2]

mismatch = num_timesamples % FFTLEN
if mismatch != 0:
	print('Warning: FFTLEN does not divide the amount of time samples')
	print(str(mismatch)+' time samples at the end will be dropped')
kept_samples = num_timesamples- mismatch

n=1

ints = np.float64(num_timesamples/FFTLEN)
print('With given FFTLEN, there are '+str(ints)+' spectra per polarization per coarse channel')

#calculate thresholds
print('Calculating SK thresholds...')
lt, ut = SK_thresholds(ints, N = 1, d = 1, p = 0.0013499)
print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))



print('Performing FFT...')

sk_results=[]

for i in range(num_coarsechan):
	for j in range(num_pol):
		print('Coarse Channel '+str(i))
		print('Polarization '+str(j))
		data_to_FFT = data[i,:kept_samples,j]
		data_to_FFT = data_to_FFT.reshape((FFTLEN,-1))
		s = np.abs(np.fft.fft(data_to_FFT,axis=0))**2
		print(s.shape)
		sk_spect = SK_EST(s,n,ints)
		sk_results.append(sk_spect)

sk_results = np.array(sk_results)
print('SK results shape: '+str(sk_results.shape))


np.save(sk_npy, sk_results)
print('SK spectra saved in '+str(sk_npy))
print('Data order: (Coarse) chan0,pol0 ; chan0,pol1 ; chan1,pol0 , etc...') 

print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))

plt.plot(sk_results[0,:],'b+')
plt.title('SK')
plt.plot(np.zeros(len(sk_results[0]))+ut, 'r-')
plt.plot(np.zeros(len(sk_results[0]))+lt, 'r-')
plt.plot(np.zeros(len(sk_results[0]))+1, 'b-')
plt.show()


plt.plot(sk_results[40,:],'b+')
plt.title('SK')
plt.plot(np.zeros(len(sk_results[0]))+ut, 'r-')
plt.plot(np.zeros(len(sk_results[0]))+lt, 'r-')
plt.plot(np.zeros(len(sk_results[0]))+1, 'b-')
plt.show()



print('Done!')












