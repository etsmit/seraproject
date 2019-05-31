#Specific program for executing SK on an entire J1713_0747 guppi file
#Lots of things will be hardcoded
#pretty much hardcoded version of guppi_SK_time


import os,sys
import numpy as np
import commands
import time

from SK_in_Python import *

my_dir = '/home/scratch/esmith/RFI_MIT/'

blocks = commands.getoutput('ls '+my_dir+'J1713+0747').split('\n')


FFTLEN = 512
SK_ints = 180

print(blocks)


#--------------------------------------
# Fun
#--------------------------------------

start = time.time()


for block in blocks:
	print('Opening file: '+block)
	data= np.load(my_dir+'J1713+0747/'+block)
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

	sk_npy = 'SK_'+block
	np.save(sk_npy, tot_sk_results)
	print('SK spectra saved in '+str(sk_npy))
	print('Data order(not including time axis): (Coarse) chan0,pol0 ; chan0,pol1 ; chan1,pol0 , etc...') 

end = time.time()
print('Entire program took '+str(end-start))

#calculate thresholds
print('Calculating SK thresholds...')
lt, ut = SK_thresholds(SK_ints, N = 1, d = 1, p = 0.0013499)

print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))

print('Done!')



