#--------------------------------------------------
#btlraw_SK.py
#Opens one output npy file from btl_to_raw.npy and performs SK 
#Takes two inputs:
#1: input filename - block style from guppiraw.py
#2: npy file to save to
#3: SK_ints - number of integrations to do SK on at once - is 'M' in the formula
#--------------------------------------------------

#Imports
import numpy as np
import os,sys
import matplotlib.pyplot as plt

import scipy as sp
import scipy.optimize
import scipy.special

from SK_in_Python import *

#--------------------------------------
# Inputs
#--------------------------------------

my_dir = '/home/scratch/esmith/RFI_MIT/'

#pulls from my scratch directory if full path not given
if sys.argv[1][0] != '/':
	inputFileName = my_dir + sys.argv[1]
else:
	inputFileName = sys.argv[1]

#same for  output destination
if sys.argv[2][0] != '/':
	base = my_dir + sys.argv[2]
else:
	base = sys.argv[2]

n=1

SK_ints = int(sys.argv[3])

sk_npy = base+'_SK.npy'
flags_npy = base+'_flags.npy'


#--------------------------------------
# Fun
#--------------------------------------


print('Opening file: '+inputFileName)
data= np.load(inputFileName)
print('Data shape: '+str(data.shape))
print('#--------------------------------------')



num_coarsechan = data.shape[0]
num_timesamples= data.shape[1]
# ^^^ FYI these are time samples of voltage corresponding to a certain frequency
# See the notebook drawing on pg 23
# FFT has already happened in the roaches
num_pol = data.shape[2]



#Check to see if SK_ints divides the total amount of data points
mismatch = num_timesamples % SK_ints
if mismatch != 0:
	print('Warning: SK_ints does not divide the amount of time samples')
	print(str(mismatch)+' time samples at the end will be dropped')
kept_samples = num_timesamples- mismatch

print('There are {} time samples and you inputted {} as m'.format(num_timesamples,SK_ints))
SK_timebins = kept_samples/SK_ints
print('Leading to '+str(SK_timebins)+' SK time bins')



kept_data = data[:,:kept_samples,:]



#calculate thresholds
print('Calculating SK thresholds...')
lt, ut = SK_thresholds(SK_ints, N = 1, d = 1, p = 0.0013499)
print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))





sk=[]


for j in range(num_pol):
	print('Polarization '+str(j))
	for k in range(SK_timebins):

		#take the stream of correct data
		start = k*SK_ints
		end = (k+1)*SK_ints
		#print('Start: {}   End: {}'.format(start,end))
		data_to_SK = data[:,start:end,j]

		#data_to_SK = data_to_SK.reshape((SK_ints,-1))#reshape to 2D array
		data_to_SK = np.abs(data_to_SK)**2#abs value and square
		#print(data_to_SK.shape)

		sk_spect = SK_EST(data_to_SK,n,SK_ints)
		sk.append(sk_spect)

sk = np.array(sk)
print('SK results shape: '+str(sk.shape))



#save results
np.save(sk_npy, sk)
print('SK spectra saved in '+str(sk_npy))

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


#look at every data point
for i in range(pols):
	for j in range(SK_timebins):
			
		#is the datapoint outside the threshold?
		if (sk[i,j] < lt) or (sk[i,j] > ut):
			flagged_pts += 1
			flags[i,j] = 1

flagged_percent = (float(flagged_pts)/tot_points)*100
print(str(flagged_pts)+' datapoints were flagged out of '+str(tot_points))
print(str(flagged_percent)+'% of data outside acceptable ranges')

np.save(flags_npy,flags)
print('Flags file saved to '+flags_npy)


print('Done!')

print('Done!')












