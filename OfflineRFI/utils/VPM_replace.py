#guppi replace
#use SK spectrum result array and a given sigma threshold to replace data on a stricter/looser basis
#INPUTS:
#1: guppi raw file to replace data in
#2: SK npy array corresponding to a certain SK_int - pol1
#3: SK npy array corresponding to a certain SK_int - pol2
#4: base filenames to save flags to (output raw data filename generated from input filename)
#5: SK_ints
#6: replacement method ('zeros', 'stats', or 'previousgood')
#7: sigma thresholding value - default is 3.0 - corresponds to PFA of 0.0013499/1



#Imports
import numpy as np
import os,sys
import matplotlib.pyplot as plt

import scipy as sp
import scipy.optimize
import scipy.special
import math as math

#import commands
import time

from blimpy import GuppiRaw

from SK_in_Python import *

#--------------------------------------
# Inputs
#--------------------------------------

#in_dir = '/export/home/ptcs/scratch/raw_RFI_data/'#assuming maxwell
in_dir = '/lustre/pulsar/users/rlynch/RFI_Mitigation/'#assuming lustre access machines
my_dir = '/home/scratch/esmith/RFI_MIT/testing/'#to save (not data) results to
#out_dir = '/export/home/ptcs/scratch/raw_RFI_data/gpu1/evan_testing/'#copies to a folder on maxwell (new ptcs)
out_dir = my_dir
in_dir = my_dir

#input raw file
#pulls from in_dir directory if full path not given
if sys.argv[1][0] != '/':
	infile = in_dir + sys.argv[1]
else:
	infile = sys.argv[1]

#input SKp1 file
#pulls from my_dir directory if full path not given
if sys.argv[1][0] != '/':
	sk_in0 = my_dir + sys.argv[2]
else:
	sk_in0 = sys.argv[2]


#input SKp2 file
#pulls from my_dir directory if full path not given
if sys.argv[1][0] != '/':
	sk_in1 = my_dir + sys.argv[3]
else:
	sk_in1 = sys.argv[3]


#base for filenames to save to (suggest source and scan number)
if sys.argv[2][0] != '/':
	base = my_dir + sys.argv[4]
else:
	base = sys.argv[4]



#number of data points to perform SK on at once/average together for spectrogram
SK_ints = int(sys.argv[5])
#FYI 1032704 (length of each block) has prime divisors (2**9) and 2017

#replacement method
#can be 'zeros','previousgood','stats' (no quotes)
method = sys.argv[6]
if not method_check(method):
	print('Incorrect replacement method - should be one of "zeros","previousgood","stats"')
	quit()

sigma = float(sys.argv[7])


#--------------------------------------
# Inits
#--------------------------------------

n=1


#threshold calc from sigma
#defined by symmetric normal distribution
SK_p = (1-scipy.special.erf(sigma/math.sqrt(2))) / 2
print('Probability of false alarm: {}'.format(SK_p))

#calculate thresholds
print('Calculating SK thresholds...')
lt, ut = SK_thresholds(SK_ints, N = 1, d = 1, p = SK_p)
print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))


#init copy of file for replaced data
print('Getting output datafile ready...')
outfile = infile[:-4]+'_'+method+'_m'+str(SK_ints)+'s'+str(sigma)+infile[-4:]

print('Loading in SK plots...')
sk_in0 = np.load(sk_in0)
sk_in1 = np.load(sk_in1)



#--------------------------------------
# Fun
#--------------------------------------


start_time = time.time()

os.system('ps')
print('Saving replaced data to '+outfile)
#os.system('rm '+outfile)
os.system('cp '+infile+' '+outfile)

if infile == outfile:
	print('ERROR: input and output files are the same.')
	sys.exit()


#load file and copy
print('Opening file: '+infile)
rawFile = GuppiRaw(infile)
print('Loading copy...')
#assuming python2 here
out_rawFile = open(outfile,'rb+')


numblocks = rawFile.find_n_data_blocks()
print('File has '+str(numblocks)+' data blocks')

flagged_pts_p1=0
flagged_pts_p2=0

flags_npy_p1 = base+'_s'+str(sigma)+'_m'+str(SK_ints)+'_flags_p1.npy'
flags_npy_p2 = base+'_s'+str(sigma)+'_m'+str(SK_ints)+'_flags_p2.npy'

flags_p1 = []
flags_p2 = []


for block in range(numblocks):
	print('#--------------------------------------')
	print('Block: '+str(block))
	if block == 0:
		header,headersize = rawFile.read_header()
		print('Header size: {} bytes'.format(headersize))
	header,data = rawFile.read_next_data_block()
	#outdata = np.array(data)
	#init replacement data and flag chunks for replacing
	repl_chunk_p1=[]
	repl_chunk_p2=[]

	out_rawFile.seek(headersize,1)

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
	SK_timebins = int(kept_samples/SK_ints)
	print('Leading to '+str(SK_timebins)+' SK time bins')

	if mismatch != 0:
		data = data[:,:kept_samples,:]


	#replacing
	for j in range(num_pol):
		if j:
			sk_arr = sk_in0
		else:
			sk_arr = sk_in1
		flagged_pts=0
		print('Polarization '+str(j))
		for k in range(SK_timebins):

			#init flag chunk
			flag_spec = np.zeros(num_coarsechan,dtype=np.int8)

			#find SK spect
			sk_spect = sk_arr[k,:]

			#flag
			for chan in range(num_coarsechan):
				#is the datapoint outside the threshold?
				if (sk_spect[chan] < lt) or (sk_spect[chan] > ut):
					flag_spec[chan] = 1
					flagged_pts += 1		

			#append to results
			if j:
				flags_p2.append(flag_spec)
				repl_chunk_p2.append(flag_spec)
				flagged_pts_p2 += flagged_pts
			else:
				flags_p1.append(flag_spec)
				repl_chunk_p1.append(flag_spec)
				flagged_pts_p1 += flagged_pts

	#Replace data
	print('Calculations complete...')
	print('Replacing Data...')

	#need to have an array here per block, but also continue appending to the list
	#transpose is to match the dimensions to the original data
	#repl_p1 = np.transpose(np.array(flags_p1))	
	#repl_p2 = np.transpose(np.array(flags_p2))

	#need to have an array here per block, but also continue appending to the list
	#transpose is to match the dimensions to the original data
	repl_chunk_p1 = np.transpose(np.array(repl_chunk_p1))	
	repl_chunk_p2 = np.transpose(np.array(repl_chunk_p2))

	if method == 'zeros':
		#replace data with zeros
		data[:,:,0] = repl_zeros(data[:,:,0],repl_chunk_p1,SK_ints)
		data[:,:,1] = repl_zeros(data[:,:,1],repl_chunk_p2,SK_ints)

	if method == 'previousgood':
		#replace data with previous (or next) good
		data[:,:,0] = previous_good(data[:,:,0],repl_chunk_p1,SK_ints)
		data[:,:,1] = previous_good(data[:,:,1],repl_chunk_p2,SK_ints)

	if method == 'stats':
		#replace data with statistical noise derived from good datapoints
		data[:,:,0] = statistical_noise(data[:,:,0],repl_chunk_p1,SK_ints)
		data[:,:,1] = statistical_noise(data[:,:,1],repl_chunk_p2,SK_ints)

	#Write back to block
	print('Re-formatting data and writing back to file...')
	data = guppi_format(data)
	print('Writing to file...')
	out_rawFile.write(data.tostring())


#save flags results
flags_p1 = np.array(flags_p1)
flags_p2 = np.array(flags_p2)

np.save(flags_npy_p1,flags_p1)
np.save(flags_npy_p2,flags_p2)
print('Flags file saved to {} and {}'.format(flags_npy_p1,flags_npy_p2))

#thresholds again
print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))

tot_points = flags_p1.size

print('Pol0: '+str(flagged_pts_p1)+' datapoints were flagged out of '+str(tot_points))
flagged_percent = (float(flagged_pts_p1)/tot_points)*100
print('Pol0: '+str(flagged_percent)+'% of data outside acceptable ranges')

print('Pol1: '+str(flagged_pts_p2)+' datapoints were flagged out of '+str(tot_points))
flagged_percent = (float(flagged_pts_p2)/tot_points)*100
print('Pol1: '+str(flagged_percent)+'% of data outside acceptable ranges')

print('Saved replaced data to '+outfile)


end_time = time.time()
elapsed = float(end_time-start_time)/60 

print('Program took {} minutes'.format(elapsed))

print('Done!')



























