#easy_replace.py
#takes two flag files and a raw (for both polarizations) and replaces data


import numpy as np
import os,sys
import matplotlib.pyplot as plt

import scipy as sp
import scipy.optimize
import scipy.special
import math as math

import commands
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
out_dir = in_dir

#input raw file
#pulls from in_dir directory if full path not given
if sys.argv[1][0] != '/':
	infile = in_dir + sys.argv[1]
else:
	infile = sys.argv[1]

#input flagp1 file
#pulls from my_dir directory if full path not given
if sys.argv[1][0] != '/':
	flag_in0 = my_dir + sys.argv[2]
else:
	flag_in0 = sys.argv[2]

#input flagp2 file
#pulls from my_dir directory if full path not given
if sys.argv[1][0] != '/':
	flag_in1 = my_dir + sys.argv[3]
else:
	flag_in1 = sys.argv[3]



#number of data points to perform SK on at once/average together for spectrogram
SK_ints = int(sys.argv[4])
#FYI 1032704 (length of each block) has prime divisors (2**9) and 2017

#replacement method
#can be 'zeros','previousgood','stats' (no quotes)
method = sys.argv[5]
if not method_check(method):
	print('Incorrect replacement method - should be one of "zeros","previousgood","stats"')
	quit()

sigma = float(sys.argv[6])

#--------------------------------------
# Inits
#--------------------------------------

n=1


#init copy of file for replaced data
print('Getting output datafile ready...')
outfile = infile[:-4]+'_'+method+'_m'+str(SK_ints)+'s'+str(sigma)+infile[-4:]

print('Loading in SK plots...')
flag_in0 = np.load(flag_in0)
flag_in1 = np.load(flag_in1)

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
out_rawFile = open(outfile,'r+')


numblocks = rawFile.find_n_data_blocks()
print('File has '+str(numblocks)+' data blocks')





for block in range(numblocks):
	print('#--------------------------------------')
	print('Block: '+str(block))
	if block == 0:
		header,headersize = rawFile.read_header()
		print('Header size: {} bytes'.format(headersize))
	header,data = rawFile.read_next_data_block()



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
	SK_timebins = kept_samples/SK_ints
	print('Leading to '+str(SK_timebins)+' SK time bins')

	#splice out part of flag array corresponding to current block
	start = block*SK_timebins
	end = (block+1)*SK_timebins
	repl_chunk_p1 = flag_in0[start:end,:]
	repl_chunk_p2 = flag_in1[start:end,:]

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


tot_points = flags_p1.size

print('Pol0: '+str(np.count_nonzero(flag_in0))+' datapoints were flagged out of '+str(tot_points))
flagged_percent = (float(np.count_nonzero(flag_in0))/tot_points)*100
print('Pol0: '+str(flagged_percent)+'% of data outside acceptable ranges')

print('Pol1: '+str(np.count_nonzero(flag_in0))+' datapoints were flagged out of '+str(tot_points))
flagged_percent = (float(np.count_nonzero(flag_in0))/tot_points)*100
print('Pol1: '+str(flagged_percent)+'% of data outside acceptable ranges')

print('Saved replaced data to '+outfile)


end_time = time.time()
elapsed = float(end_time-start_time)/60 

print('Program took {} minutes'.format(elapsed))

print('Done!')













