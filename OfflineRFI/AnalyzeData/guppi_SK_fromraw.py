#--------------------------------------------------
#btlraw_SK.py
#Opens GUPPI raw file, performs SK, gibs flags, no replacement
#Takes four inputs:
#1: Guppi raw file
#2: npy file to save to
#3: True/False save raw data to block style numpy files
#4: SK_ints - number of integrations to do SK on at once - is 'M' in the formula

#Note - for 128 4GB blocks and SK_ints=512, expect ~16-17GB max memory usage.
#Lowering SK_ints increases memory usage slightly less than linearly
#--------------------------------------------------


#Imports
import numpy as np
import os,sys
import matplotlib.pyplot as plt

import scipy as sp
import scipy.optimize
import scipy.special

import commands
import time

from blimpy import GuppiRaw

from SK_in_Python import *

#--------------------------------------
# Inputs
#--------------------------------------

my_dir = '/home/scratch/esmith/RFI_MIT/'


#input directory
#pulls from my scratch directory if full path not given
if sys.argv[1][0] != '/':
	infile = my_dir + sys.argv[1]
else:
	infile = sys.argv[1]


#base for filenames to save to (suggest source and scan number)
if sys.argv[2][0] != '/':
	base = my_dir + sys.argv[2]
else:
	base = sys.argv[2]

#save raw data to npy files?
rawdata = sys.argv[3]


#number of data points to perform SK on at once/average together for spectrogram
SK_ints = int(sys.argv[4])
#FYI 1032704 (length of each block) has prime divisors (2**9) and 2017


#--------------------------------------
# Inits
#--------------------------------------

n=1

#filenames to save to
#'p' stands for polarization
sk_npy_p1 = base+'_SK_p1.npy'
sk_npy_p2 = base+'_SK_p2.npy'
flags_npy_p1 = base+'_flags_p1.npy'
flags_npy_p2 = base+'_flags_p2.npy'
spect_npy_p1 = base+'_spect_p1.npy'
spect_npy_p2 = base+'_spect_p2.npy'

#calculate thresholds
print('Calculating SK thresholds...')
lt, ut = SK_thresholds(SK_ints, N = 1, d = 1, p = 0.0013499)
print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))


#array to hold SK results
sk_p1=[]
sk_p2=[]

#array to hold spectrum results
spect_results_p1 = []
spect_results_p2 = []

#flags results get initialization in APPLY FLAGS section (after data shape is determined)

#save raw data
if rawdata == 'True':
	rawdata = True
if rawdata == 'False':
	rawdata = False

if rawdata:
	print('Saving raw data to npy block style files')
	os.system('mkdir '+base)


#--------------------------------------
# Fun
#--------------------------------------


start_time = time.time()

#load file
print('Opening file: '+infile)
rawFile = GuppiRaw(infile)

numblocks = rawFile.find_n_data_blocks()
print('File has '+str(numblocks)+' data blocks')


for block in range(numblocks):
	print('#--------------------------------------')
	print('Block: '+str(block))
	header,data = rawFile.read_next_data_block(block)
	
	#print header for the first block
	if block == 0:
		print('Datatype: '+str(type(data[0,0,0])))
		for line in header:
			print(line+':  '+str(header[line]))

	print('Data shape: '+str(data.shape))

	#save raw data
	if rawdata:
		#pad number to three digits
		block = str(block)
		if blockNumber <10:
			block = '0'+block
		if blockNumber <100:
			block = '0'+block

		save_fname = base+'_block'+block+'.npy'
		np.save(base+'/'+save_fname,data)
		print('Saved under '+outdir+save_fname)



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
		print('Recommend SK_ints is a power of 2 up to 512 or 2017 for 1032704 datapoints')
	kept_samples = num_timesamples- mismatch
	
	print('There are {} time samples and you inputted {} as m'.format(num_timesamples,SK_ints))
	SK_timebins = kept_samples/SK_ints
	print('Leading to '+str(SK_timebins)+' SK time bins')


	kept_data = data[:,:kept_samples,:]


	#Calculations
	for j in range(num_pol):
		print('Polarization '+str(j))
		for k in range(SK_timebins):
	
			#take the stream of correct data
			start = k*SK_ints
			end = (k+1)*SK_ints
			data_chunk = data[:,start:end,j]

			#square it
			data_chunk = np.abs(data_chunk)**2#abs value and square

			#perform SK
			sk_spect = SK_EST(data_chunk,n,SK_ints)
			#average power spectrum
			spectrum = np.average(data_chunk,axis=1)

			#append to results
			if j:
				sk_p2.append(sk_spect)
				spect_results_p2.append(spectrum)
			else:
				sk_p1.append(sk_spect)
				spect_results_p1.append(spectrum)



#save SK results
sk_p1 = np.array(sk_p1)
sk_p2 = np.array(sk_p2)
print('Final results shape: '+str(sk_p1.shape))

np.save(sk_npy_p1, sk_p1)
np.save(sk_npy_p2, sk_p2)
print('SK spectra saved in {} and {}'.format(sk_npy_p1,sk_npy_p2))


#save spectrum results
spect_results_p1 = np.array(spect_results_p1)
spect_results_p2 = np.array(spect_results_p2)

np.save(spect_npy_p1, spect_results_p1)
np.save(spect_npy_p2, spect_results_p2)
print('Spectra saved in {} and {}'.format(spect_npy_p1,spect_npy_p2))

#thresholds again
print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))




#------------------------------------------------
# APPLY FLAGS
# modified from SK_stats_time.py
#------------------------------------------------

print('Flagging now...')

flags_p1 = np.zeros(sk_p1.shape)
flags_p2 = np.zeros(sk_p2.shape)
tot_points = sk_p1.size



#look at every data point
for i in range(num_pol):
	flagged_pts = 0
	for j in range(SK_timebins*numblocks):
		for k in range(num_coarsechan):
			if i:
				#print('Polarization 2')
				#is the datapoint outside the threshold?
				if (sk_p2[j,k] < lt) or (sk_p2[j,k] > ut):
					flagged_pts += 1
					flags_p2[j,k] = 1
			else:
				#print('Polarization 1')
				#is the datapoint outside the threshold?
				if (sk_p1[j,k] < lt) or (sk_p1[j,k] > ut):
					flagged_pts += 1
					flags_p1[j,k] = 1


	flagged_percent = (float(flagged_pts)/tot_points)*100
	print(str(flagged_pts)+' datapoints were flagged out of '+str(tot_points))
	print(str(flagged_percent)+'% of data outside acceptable ranges')



np.save(flags_npy_p1,flags_p1)
np.save(flags_npy_p2,flags_p2)
print('Flags file saved to {} and {}'.format(flags_npy_p1,flags_npy_p2))

end_time = time.time()
elapsed = float(end_time-start_time)/60 

print('Program took {} minutes'.format(elapsed))

print('Done!')












