#--------------------------------------------------
"""
detectRFI_VPM.py


Main program for detecting/excising RFI in seraproj
github.com/etsmit/seraproj

 - Opens GUPPI/VPM raw file 
 - performs SK
 - gibs flags
 - replaces flagged data
 - gibs copy of data with flagged data replaced (optional 

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
  -rfi {SKurtosis,SEntropy}
                        String. Required. RFI detection method desired.
  -m SK_INTS            Integer. Required. "M" in the SK equation. Number of
                        data points to perform SK on at once/average together
                        for spectrogram. ex. 1032704 (length of each block)
                        has prime divisors (2**9) and 2017. Default 512.
  -r {zeros,previousgood,stats}
                        String. Required. Replacement method of flagged data
                        in output raw data file. Can be
                        "zeros","previousgood", or "stats"
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

from RFI_detect import *
from RFI_support import *

#--------------------------------------
# Inputs
#--------------------------------------

#in_dir = '/export/home/ptcs/scratch/raw_RFI_data/'#assuming maxwell
#in_dir = '/lustre/pulsar/users/rlynch/RFI_Mitigation/'#assuming lustre access machines
in_dir = '/data/rfimit/unmitigated/rawdata/'#leibniz
#my_dir = '/home/scratch/esmith/RFI_MIT/testing/'#to save (not data) results to
#out_dir = '/export/home/ptcs/scratch/raw_RFI_data/gpu1/evan_testing/'#copies to a folder on maxwell (new ptcs)
my_dir = '/data/scratch/Spring2020/'
out_dir = my_dir 


#argparse parsing
parser = argparse.ArgumentParser(description="""function description""")

#input file
parser.add_argument('-i',dest='infile',type=str,required=True,help='String. Required. Name of input filename. Automatically pulls from standard data directory. If leading "/" given, pulls from given directory')

#RFI detection method
parser.add_argument('-rfi',dest='RFI',type=str,required=True,choices=['SKurtosis','SEntropy'],default='SKurtosis',help='String. Required. RFI detection method desired.')

#SK integrations. 'M' in the SK equation. Number of data points to perform SK on at once/average together for spectrogram. FYI 1032704 (length of each block) has prime divisors (2**9) and 2017.
parser.add_argument('-m',dest='SK_ints',type=int,required=True,default=512,help='Integer. Required. "M" in the SK equation. Number of data points to perform SK on at once/average together for spectrogram. ex. 1032704 (length of each block) has prime divisors (2**9) and 2017. Default 512.')


#replacement method
parser.add_argument('-r',dest='method',type=str,choices=['zeros','previousgood','stats'], required=True,default='zeros',help='String. Required. Replacement method of flagged data in output raw data file. Can be "zeros","previousgood", or "stats"')



#sigma thresholding
parser.add_argument('-s',dest='sigma',type=float,default=3.0,help='Float. Sigma thresholding value. Default of 3.0 gives probability of false alarm 0.001349')

#number of inside accumulations, 'N' in the SK equation
parser.add_argument('-n',dest='n',type=int,default=1,help='Integer. Number of inside accumulations, "N" in the SK equation. Default 1.')

#vegas spectral line file? needs new data directory and session
parser.add_argument('-v',dest='vegas_dir',type=str,default='0',help='If inputting a VEGAS spectral line mode file, enter AGBT19B_335 session number (1/2) and bank (C/D) ex "1D".')

#write out a whole new raw file or just get SK/accumulated spectra results
parser.add_argument('-newfile',dest='output_bool',type=bool,default=False,help='Copy the original data and output a replaced datafile. Default True. Change to False to not write out a whole new GUPPI file')

#pick d in the case that it isn't 1. Required for low-bit quantization.
#Can be found (i think) by running SK and changing d to be 1/x, where x is the center of the SK value distribution.
parser.add_argument('-d',dest='d',type=float,default=1.,help='Float. Shape parameter d. Default 1, but is different in the case of low-bit quantization. Can be found (i think) by running SK and changing d to be 1/x, where x is the center of the SK value distribution.')

#Save raw data to npy files (storage intensive, unnecessary)
parser.add_argument('-npy',dest='rawdata',type=bool,default=False,help='Boolean. True to save raw data to npy files. This is storage intensive and unnecessary since blimpy. Default is False')


#parse input variables
args = parser.parse_args()
infile = args.infile
SK_ints = args.SK_ints
method = args.method
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


base = my_dir+infile[len(in_dir):-4]

#filenames to save to
#'p' stands for polarization
sk_filename = base+'_SK_m'+str(SK_ints)+'_'+method+'_s'+str(sigma)+'.npy'
flags_filename = base+'_flags_m'+str(SK_ints)+'_'+method+'_s'+str(sigma)+'.npy'
spect_filename = base+'_spect_m'+str(SK_ints)+'_'+method+'_s'+str(sigma)+'.npy'

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

	if output_bool:
		out_rawFile.seek(headersize,1)

	num_coarsechan = data.shape[0]
	num_timesamples= data.shape[1]
	# ^^^ FYI these are time samples of voltage corresponding to a certain frequency
	# See the notebook drawing on pg 23
	# FFT has already happened in the roaches
	num_pol = data.shape[2]

	print('Data shape: '+str(data.shape))

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

	#ASSUMING NPOL = 2:

	for k in range(SK_timebins):
	
		#take the stream of correct data
		start = k*SK_ints
		end = (k+1)*SK_ints
		data_chunk = data[:,start:end,:]

		sk_spect = np.zeros((num_coarsechan,2))

		#square it
		data_chunk = np.abs(data_chunk)**2#abs value and square

		#perform SK
		sk_spect[:,0] = SK_EST(data_chunk[:,:,0],n,SK_ints,d)
		sk_spect[:,1] = SK_EST(data_chunk[:,:,1],n,SK_ints,d)
		#average power spectrum
		spectrum = np.average(data_chunk,axis=1)
		#init flag chunk
		flag_spect = np.zeros((num_coarsechan,2),dtype=np.int8)

		#flag (each pol separately, for records)
		flag_spect[sk_spect>ut] = 1
		flag_spect[sk_spect<lt] = 1

		

		#append to results
		if (k==0):
			sk_block=np.expand_dims(sk_spect,axis=2)
			spect_block=np.expand_dims(spectrum,axis=2)
			flags_block = np.expand_dims(flag_spect,axis=2)

		else:
			sk_block=np.c_[sk_block,np.expand_dims(sk_spect,axis=2)]
			spect_block=np.c_[spect_block,np.expand_dims(spectrum,axis=2)]
			flags_block = np.c_[flags_block,np.expand_dims(flag_spect,axis=2)]


	if (block==0):
		sk_all = sk_block
		spect_all = spect_block
		flags_all = flags_block
	else:
		sk_all = np.c_[sk_all,sk_block]
		spect_all = np.c_[spect_all,spect_block]
		flags_all = np.c_[flags_all,flags_block]


	#Replace data
	print('Calculations complete...')
	print('Replacing Data...')
	
	repl_chunk=np.transpose(flags_block,(0,2,1))
	#now flag shape is (chan,spectra,pol)
	#apply union of flags
	repl_chunk[:,:,0][repl_chunk[:,:,1]==1]=1
	repl_chunk[:,:,1][repl_chunk[:,:,0]==1]=1
	
	extend = np.ones((1,SK_ints,1))

	#extend flagging array
	repl_chunk = np.kron(repl_chunk,extend)


	if method == 'zeros':
		#replace data with zeros
		data = repl_zeros(data,repl_chunk)

	if method == 'previousgood':
		#replace data with previous (or next) good
		data = previous_good(data,repl_chunk,SK_ints)

	if method == 'stats':
		#replace data with statistical noise derived from good datapoints
		data = statistical_noise(data,repl_chunk,SK_ints)

	#Write back to block
	if output_bool:
		print('Re-formatting data and writing back to file...')
		data = guppi_format(data)
		out_rawFile.write(data.tostring())



#save SK results
sk_all = np.transpose(sk_all,(0,2,1))
print('Final results shape: '+str(sk_all.shape))

np.save(sk_filename, sk_all)
print('SK spectra saved in {}'.format(sk_filename))


#save spectrum results
spect_all = np.transpose(spect_all,(0,2,1))
np.save(spect_filename, spect_all)
print('Spectra saved in {}'.format(spect_filename))


#save flags results
flags_all = np.transpose(flags_all,(0,2,1))
np.save(flags_filename,flags_all)
print('Flags file saved to {}'.format(flags_filename))

#thresholds again
print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))

tot_points = flags_all[:,:,1].size
flagged_pts_p1 = np.count_nonzero(flags_all[:,:,0])
flagged_pts_p2 = np.count_nonzero(flags_all[:,:,1])

print('Pol0: '+str(flagged_pts_p1)+' datapoints were flagged out of '+str(tot_points))
flagged_percent = (float(flagged_pts_p1)/tot_points)*100
print('Pol0: '+str(flagged_percent)+'% of data outside acceptable ranges')

print('Pol1: '+str(flagged_pts_p2)+' datapoints were flagged out of '+str(tot_points))
flagged_percent = (float(flagged_pts_p2)/tot_points)*100
print('Pol1: '+str(flagged_percent)+'% of data outside acceptable ranges')

tot_points = flags_all.size
flagged_pts_all = np.count_nonzero(flags_all)

print('Union of flags: {}% of data flagged'.format((100.*flagged_pts_p1)/tot_points))




print('Saved replaced data to '+outfile)


end_time = time.time()
elapsed = float(end_time-start_time)/60 

print('Program took {} minutes'.format(elapsed))

print('Done!')
