#--------------------------------------------------
"""
RegenerateSpect.py


MaJust makes the averaged spectrogram of a mitigated datafile to compare power and 
data replacement effectiveness
github.com/etsmit/seraproj

 - Opens GUPPI/VPM raw file 
 - Averages every M spectra and makes output npy array

Use instructions:

 - python3.6.5
 - use /users/esmith/.conda/envs/py365 conda environment on green bank machine
   - (desired environment psrenv doesn't import blimpy 9/28/20)
 - type ' -h' to see help message

Inputs
------------
  -h, --help            show this help message and exit
  -i INFILE             String. Required. Name of input filename.
                        Automatically pulls from standard data directory. If
                        leading "/" given, pulls from given directory
  -m SK_INTS            Integer. Required. "M" in the SK equation. Number of
                        data points to perform SK on at once/average together
                        for spectrogram. ex. 1032704 (length of each block)
                        has prime divisors (2**9) and 2017. Default 512.
  -s SIGMA              Float. Sigma thresholding value. Default of 3.0 gives
                        probability of false alarm 0.001349
  -n N                  Integer. Number of inside accumulations, "N" in the SK
                        equation. Default 1.
  -v VEGAS_DIR          If inputting a VEGAS spectral line mode file, enter
                        AGBT19B_335 session number (1/2) and bank (C/D) ex
                        "1D".
  -newfile OUTPUT_BOOL  Copy the original data and output a replaced datafile.
                        Default True. Change to False to not write out a whole
                        new GUPPI file
  -d D                  Float. Shape parameter d. Default 1, but is different
                        in the case of low-bit quantization. Can be found (i
                        think) by running SK and changing d to be 1/x, where x
                        is the center of the SK value distribution.
  -npy RAWDATA          Boolean. True to save raw data to npy files. This is
                        storage intensive and unnecessary since blimpy.
                        Default is False




#Note - for 128 4GB blocks and SK_ints=512, expect ~16-17GB max memory usage.
#Lowering SK_ints increases memory usage slightly less than linearly
#Assumes two polarizations
#see RFI_detect.py and RFI_support.py for functions used
"""
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

from RFI_detection import *
from RFI_support import *

#--------------------------------------
# Inputs
#--------------------------------------

#in_dir = '/export/home/ptcs/scratch/raw_RFI_data/'#assuming maxwell
#in_dir = '/lustre/pulsar/users/rlynch/RFI_Mitigation/'#assuming lustre access machines
in_dir = '/data/rfimit/unmitigated/rawdata/'#leibniz
npy_dir = '/home/scratch/esmith/RFI_MIT/npy_test/'#to save (not data) results to
#out_dir = '/export/home/ptcs/scratch/raw_RFI_data/gpu1/evan_testing/'#copies to a folder on maxwell (new ptcs)
my_dir = '/data/scratch/Summer2022/'
out_dir = my_dir 


#argparse parsing
parser = argparse.ArgumentParser(description="""function description""")

#input file
parser.add_argument('-i',dest='infile',type=str,required=True,help='String. Required. Name of input filename. Automatically pulls from standard data directory. If leading "/" given, pulls from given directory')



#SK integrations. 'M' in the SK equation. Number of data points to perform SK on at once/average together for spectrogram. FYI 1032704 (length of each block) has prime divisors (2**9) and 2017.
parser.add_argument('-m',dest='SK_ints',type=int,required=True,default=512,help='Integer. Required. "M" in the SK equation. Number of data points to perform SK on at once/average together for spectrogram. ex. 1032704 (length of each block) has prime divisors (2**9) and 2017. Default 512.')




#----

#sigma thresholding
parser.add_argument('-s',dest='sigma',type=float,default=3.0,help='Float. Sigma thresholding value. Default of 3.0 gives probability of false alarm 0.001349')

#number of inside accumulations, 'N' in the SK equation
parser.add_argument('-n',dest='n',type=int,default=1,help='Integer. Number of inside accumulations, "N" in the SK equation. Default 1.')

#vegas spectral line file? needs new data directory and session
parser.add_argument('-v',dest='vegas_dir',type=str,default='0',help='If inputting a VEGAS spectral line mode file, enter AGBT19B_335 session number (1/2) and bank (C/D) ex "1D".')

#write out a whole new raw file or just get SK/accumulated spectra results
parser.add_argument('-newfile',dest='output_bool',type=bool,default=True,help='Copy the original data and output a replaced datafile. Default True. Change to False to not write out a whole new GUPPI file')

#pick d in the case that it isn't 1. Required for low-bit quantization.
#Can be found (i think) by running SK and changing d to be 1/x, where x is the center of the SK value distribution.
parser.add_argument('-d',dest='d',type=float,default=1.,help='Float. Shape parameter d. Default 1, but is different in the case of low-bit quantization. Can be found (i think) by running SK and changing d to be 1/x, where x is the center of the SK value distribution.')

#Save raw data to npy files (storage intensive, unnecessary)
parser.add_argument('-npy',dest='rawdata',type=bool,default=False,help='Boolean. True to save raw data to npy files. This is storage intensive and unnecessary since blimpy. Default is False')


#parse input variables
args = parser.parse_args()
infile = args.infile
SK_ints = args.SK_ints
rawdata = args.rawdata
sigma = args.sigma
n = args.n
v_s = args.vegas_dir[0]
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


#base = npy_dir+infile[len(in_dir):-4]
base = my_dir+infile[len(in_dir):-4]

#filenames to save to
#'p' stands for polarization
spect_filename = base+'_regen_m'+str(SK_ints)+'.npy'

print(spect_filename)

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



#--------------------------------------
# Fun
#--------------------------------------


start_time = time.time()



#os.system('rm '+outfile)
#if output_bool:
#	print('Saving replaced data to '+outfile)
#	os.system('cp '+infile+' '+outfile)
#	out_rawFile = open(outfile,'rb+')

#load file and copy
print('Opening file: '+infile)
rawFile = GuppiRaw(infile)
print('Loading copy...')
#assuming python3 here



numblocks = rawFile.find_n_data_blocks()
print('File has '+str(numblocks)+' data blocks')



for block in range(numblocks):
	print('------------------------------------------')
	print('Block: '+str(block))
	if block == 0:
		header,headersize = rawFile.read_header()
		print('Header size: {} bytes'.format(headersize))
	header,data = rawFile.read_next_data_block()


	


	#print header for the first block
	if block == 0:
		print('Datatype: '+str(type(data[0,0,0])))
		for line in header:
			print(line+':  '+str(header[line]))

	#if output_bool:
	#	out_rawFile.seek(headersize,1)

	num_coarsechan = data.shape[0]
	num_timesamples= data.shape[1]
	# ^^^ FYI these are time samples of voltage corresponding to a certain frequency
	# See the notebook drawing on pg 23
	# FFT has already happened in the roaches
	num_pol = data.shape[2]

	print('Data shape: {} || block size: {}'.format(data.shape,data.nbytes))

	#save raw data
	if rawdata:
		#pad number to three digits
		block_fname = str(block).zfill(3)

		save_fname = base+'_block'+block_fname+'.npy'
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

	#ASSUMING NPOL = 2:

	for k in range(SK_timebins):
	
		#take the stream of correct data
		start = k*SK_ints
		end = (k+1)*SK_ints
		data_chunk = data[:,start:end,:]

		#square it
		data_chunk = np.abs(data_chunk)**2#abs value and square

		#perform RFI detection



		#average power spectrum
		spectrum = np.average(data_chunk,axis=1)


		

		#append to results
		if (k==0):
			spect_block=np.expand_dims(spectrum,axis=2)


		else:
			spect_block=np.c_[spect_block,np.expand_dims(spectrum,axis=2)]

	#adj_chan flagging here
	#flags_block = adj_chan_skflags(spect_block,flags_block,sk_block,1,3)



	#print(block)
	if (block==0):
		spect_all = spect_block
	else:
		spect_all = np.c_[spect_all,spect_block]












#save spectrum results
spect_all = np.transpose(spect_all,(0,2,1))
np.save(spect_filename, spect_all)
print('Spectra saved in {}'.format(spect_filename))





end_time = time.time()
elapsed = float(end_time-start_time)/60 

print('Program took {} minutes'.format(elapsed))

print('Done!')
