#--------------------------------------------------
#py3_guppi_SK_fromraw.py
#python3 version
#Opens GUPPI raw file, performs SK, gibs flags, no replacement
#Takes four inputs:
#1: Guppi raw file
#2: base filename to save to
#3: True/False to save raw data to block style numpy files
#4: SK_ints - number of integrations to do SK on at once - is 'M' in the formula
#5: replacement method
#6: sigma thresholding value - default is 3.0 - corresponds to PFA of 0.0013499/1

#Note - for 128 4GB blocks and SK_ints=512, expect ~16-17GB max memory usage.
#Lowering SK_ints increases memory usage slightly less than linearly
#Assumes two polarizations
#see SK_in_Python.py for functions used
#--------------------------------------------------


#Imports
import numpy as np
import os,sys
import matplotlib.pyplot as plt

import scipy as sp
import scipy.optimize
import scipy.special
import math as math

import argparse

import time

from blimpy import GuppiRaw

from SK_in_Python import *

#--------------------------------------
# Inputs
#--------------------------------------

#in_dir = '/export/home/ptcs/scratch/raw_RFI_data/'#assuming maxwell
#in_dir = '/lustre/pulsar/users/rlynch/RFI_Mitigation/'#assuming lustre access machines
in_dir = '/data/rfimit/unmitigated/rawdata/'#leibniz
#my_dir = '/home/scratch/esmith/RFI_MIT/testing/'#to save (not data) results to
#out_dir = '/export/home/ptcs/scratch/raw_RFI_data/gpu1/evan_testing/'#copies to a folder on maxwell (new ptcs)
my_dir = '/data/scratch/Winter2020/'
out_dir = my_dir 


#argparse parsing
parser = argparse.ArgumentParser(description="""function description""")

#input file
parser.add_argument('-i',dest='infile',type=str,required=True,help='String. Name of input filename. Automatically pulls from standard data directory. If leading "/" given, pulls from given directory')

#base output filenames
#AUTOMATICALLY GENERATED FROM NOW ON

#Save raw data to npy files (storage intensive, unnecessary)
parser.add_argument('-npy',dest='rawdata',type=bool,default=False,help='Boolean. True to save raw data to npy files. This is storage intensive and unnecessary since blimpy. Default is False')

#SK integrations. 'M' in the SK equation. Number of data points to perform SK on at once/average together for spectrogram. FYI 1032704 (length of each block) has prime divisors (2**9) and 2017.
parser.add_argument('-m',dest='SK_ints',type=int,required=True,default=512,help='Integer. "M" in the SK equation. Number of data points to perform SK on at once/average together for spectrogram. FYI 1032704 (length of each block) has prime divisors (2**9) and 2017. Default 512.')


#replacement method
parser.add_argument('-r',dest='method',type=str,choices=['zeros','previousgood','stats'], required=True,default='zeros',help='String. Replacement method of flagged data in output raw data file. Can be "zeros","previousgood", or "stats"')

#sigma thresholding
parser.add_argument('-s',dest='sigma',type=float,default=3.0,help='Float. Sigma thresholding value. Default of 3.0 gives probability of false alarm 0.001349')

#number of inside accumulations, 'N' in the SK equation
parser.add_argument('-n',dest='n',type=int,default=1,help='Integer. Number of inside accumulations, "N" in the SK equation. Default 1.')

#vegas file? needs new data directory and session
parser.add_argument('-v',dest='vegas_dir',type=str,default='0',help='If inputting a VEGAS file, enter AGBT19B_335 session number (1/2) and bank (C/D) ex "1D".')

#write out a whole new raw file or just get SK/accumulated spectra results
parser.add_argument('-newfile',dest='output_bool',type=bool,default=False,help='Copy the original data and output a replaced datafile. Default True. Change to False to not write out a whole new GUPPI file')

#pick d in the case that it isn't 1. Required for low-bit quantization.
#Can be found (i think) by running SK and changing d to be 1/x, where x is the center of the SK value distribution.
parser.add_argument('-d',dest='d',type=float,default=1.,help='Float. Shape parameter d. Default 1, but is different in the case of low-bit quantization. Can be found (i think) by running SK and changing d to be 1/x, where x is the center of the SK value distribution.')


#parse input variables
args = parser.parse_args()
infile = args.infile
SK_ints = args.SK_ints
method = args.method
rawdata = args.rawdata
sigma = args.sigma
n = args.n
v_s = args.vegas_dir[0]
print(v_s)
if v_s != '0':
	v_b = args.vegas_dir[1]
	in_dir = in_dir+'vegas/AGBT19B_335_0'+v_s+'/VEGAS/'+v_b+'/'
output_bool = args.output_bool
d = args.d
 


#input file
#pulls from my scratch directory if full path not given
if infile[0] != '/':
	infile = in_dir + infile
else:
	in_dir = infile[:infile.rindex('/')+1]

if infile[-4:] != '.raw':
	print("WARNING input filename doesn't end in '.raw'. Auto-generated output files will have weird names.")

#--------------------------------------
# Inits
#--------------------------------------


base = my_dir+infile[len(in_dir):-4]

#filenames to save to
#'p' stands for polarization
sk_npy_p1 = base+'_SK_m'+str(SK_ints)+'_'+method+'_s'+str(sigma)+'_p1.npy'
sk_npy_p2 = base+'_SK_m'+str(SK_ints)+'_'+method+'_s'+str(sigma)+'_p2.npy'
flags_npy_p1 = base+'_flags_m'+str(SK_ints)+'_'+method+'_s'+str(sigma)+'_p1.npy'
flags_npy_p2 = base+'_flags_m'+str(SK_ints)+'_'+method+'_s'+str(sigma)+'_p2.npy'
spect_npy_p1 = base+'_spect_m'+str(SK_ints)+'_'+method+'_s'+str(sigma)+'_p1.npy'
spect_npy_p2 = base+'_spect_m'+str(SK_ints)+'_'+method+'_s'+str(sigma)+'_p2.npy'

#threshold calc from sigma
#defined by symmetric normal distribution
SK_p = (1-scipy.special.erf(sigma/math.sqrt(2))) / 2
print('Probability of false alarm: {}'.format(SK_p))

#calculate thresholds
print('Calculating SK thresholds...')
lt, ut = SK_thresholds(SK_ints, N = n, d = d, p = SK_p)
print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))



if rawdata:
	print('Saving raw data to npy block style files')

#init copy of file for replaced data
print('Getting output datafile ready...')
outfile = out_dir + infile[len(in_dir):-4]+'_'+method+'_m'+str(SK_ints)+'s'+str(sigma)+infile[-4:]



#--------------------------------------
# Fun
#--------------------------------------


start_time = time.time()



#os.system('rm '+outfile)
if output_bool:
	print('Saving replaced data to '+outfile)
	os.system('cp '+infile+' '+outfile)
	out_rawFile = open(outfile,'rb+')

#load file and copy
print('Opening file: '+infile)
rawFile = GuppiRaw(infile)
print('Loading copy...')
#assuming python3 here



numblocks = rawFile.find_n_data_blocks()
print('File has '+str(numblocks)+' data blocks')

flagged_pts_p1=0
flagged_pts_p2=0


for block in range(numblocks):
	print('------------------------------------------')
	print('Block: '+str(block))
	if block == 0:
		header,headersize = rawFile.read_header()
		print('Header size: {} bytes'.format(headersize))
	header,data = rawFile.read_next_data_block()


	#init replacement data and flag chunks for replacing
	new_block = np.zeros(data.shape)
	repl_chunk_p1=[]
	repl_chunk_p2=[]	


	#print header for the first block
	if block == 0:
		print('Datatype: '+str(type(data[0,0,0])))
		for line in header:
			print(line+':  '+str(header[line]))

	if output_bool:
		out_rawFile.seek(headersize,1)

	num_coarsechan = data.shape[0]
	num_timesamples= data.shape[1]
	# ^^^ FYI these are time samples of voltage corresponding to a certain frequency
	# See the notebook drawing on pg 23
	# FFT has already happened in the roaches
	num_pol = data.shape[2]

	print('Data shape: '+str(data.shape))

	blockNumber = block
	#save raw data
	if rawdata:
		#pad number to three digits
		block = str(block).zfill(3)

		save_fname = base+'_block'+block+'.npy'
		np.save(save_fname,data)
		#print('Saved under '+out_dir+save_fname)



	#Check to see if SK_ints divides the total amount of data points
	mismatch = num_timesamples % SK_ints
	if mismatch != 0:
		print('Warning: SK_ints does not divide the amount of time samples')
		print(str(mismatch)+' time samples at the end will be dropped')
		print('Recommend SK_ints is a power of 2 up to 512 or 2017 for 1032704 datapoints')
	kept_samples = int(num_timesamples- mismatch)
	
	print('There are {} time samples left and you inputted {} as m'.format(kept_samples,SK_ints))
	SK_timebins = int(kept_samples/SK_ints)
	print('Leading to '+str(SK_timebins)+' SK time bins')

	if mismatch != 0:
		data = data[:,:kept_samples,:]


	#Calculations
	for j in range(num_pol):
		flagged_pts=0
		#print('Polarization '+str(j))
		for k in range(SK_timebins):
	
			#take the stream of correct data
			start = k*SK_ints
			end = (k+1)*SK_ints
			data_chunk = data[:,start:end,j]

			#square it
			data_chunk = np.abs(data_chunk)**2#abs value and square

			#perform SK
			sk_spect = SK_EST(data_chunk,n,SK_ints,d)
			#average power spectrum
			spectrum = np.average(data_chunk,axis=1)
			#init flag chunk
			flag_spec = np.zeros(num_coarsechan,dtype=np.int8)

			#flag
			for chan in range(num_coarsechan):
				#is the datapoint outside the threshold?
				if (sk_spect[chan] < lt) or (sk_spect[chan] > ut):
					flag_spec[chan] = 1
					flagged_pts += 1		

			#append to results
			if (block==0):
				if j:
					sk_p2 = sk_spect
					spect_results_p2 = spectrum
					flags_p2 = flag_spec
					repl_chunk_p2.append(flag_spec)
				else:
					sk_p1 = sk_spect
					spect_results_p1 = spectrum
					flags_p1 = flag_spec
					repl_chunk_p1.append(flag_spec)
			else:
				if j:
					sk_p2 = np.c_[sk_p2,sk_spect]
					spect_results_p2 = np.c_[spect_results_p2,spectrum]
					flags_p2 = np.c_[flags_p2,flag_spec]
					repl_chunk_p2.append(flag_spec)
				else:
					sk_p1 = np.c_[sk_p1,sk_spect]
					spect_results_p1 = np.c_[spect_results_p1,spectrum]
					flags_p1 = np.c_[flags_p1,flag_spec]
					repl_chunk_p1.append(flag_spec)





	#Replace data
	print('Calculations complete...')
	print('Replacing Data...')


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
	if output_bool:
		print('Re-formatting data and writing back to file...')
		data = guppi_format(data)
		out_rawFile.write(data.tostring())



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


#save flags results
flags_p1 = np.array(flags_p1)
flags_p2 = np.array(flags_p2)

np.save(flags_npy_p1,flags_p1)
np.save(flags_npy_p2,flags_p2)
print('Flags file saved to {} and {}'.format(flags_npy_p1,flags_npy_p2))

#thresholds again
print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))

tot_points = sk_p1.size
flagged_pts_p1 = np.count_nonzero(flags_p1)
flagged_pts_p2 = np.count_nonzero(flags_p2)

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
