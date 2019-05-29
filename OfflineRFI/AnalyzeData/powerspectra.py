#creates total power spectra of guppi data
#uncalibrated for now



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
	outfile = my_dir + sys.argv[2]
else:
	outfile = sys.argv[2]



FFTLEN = float(sys.argv[3])




print('Opening file: '+inputFileName)
data= np.load(inputFileName)
print('Data shape: '+str(data.shape))
print('#--------------------------------------')


mismatch = data.shape[1] % FFTLEN
if mismatch != 0:
	print('Warning: FFTLEN does not divide the amount of time samples')
	print(str(mismatch)+' time samples at the end will be dropped')
kept_samples = data.shape[1] - mismatch



print('Performing FFT...')

spectra=[]

for i in range(data.shape[0]):
	for j in range(data.shape[2]):
		print('Coarse Channel '+str(i))
		print('Polarization '+str(j))
		data_to_FFT = data[i,:kept_samples,j]
		data_to_FFT = data_to_FFT.reshape((FFTLEN,-1))
		s = np.abs(np.fft.fft(data_to_FFT,axis=0))**2
		print(str(s.shape[1])+' spectra with '+str(s.shape[0])+' channels each')
		spectra.append(s)

spectra=np.array(spectra)

print('Saving power spectra...')
np.save(outfile,spectra)







