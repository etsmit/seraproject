#--------------------------------------------------
"""
detectRFI_VPM.py


Framework for detecting/excising RFI in seraproj
github.com/etsmit/seraproj

 - Opens GUPPI/VPM raw file 
 - ***has an open spot for RFI detection (lines 249-256)
 - gibs flags
 - replaces flagged data
 - gibs copy of data with flagged data replaced (optional) 

Use instructions:

 - python3.6.5
 - use /users/esmith/.conda/envs/py365 conda environment on green bank machine
 - or psrenv
 - type ' -h' to see help message

Inputs
------------
  -h, --help            show this help message and exit
  -i INFILE             String. Required. Name of input filename.
                        Automatically pulls from standard data directory. If
                        leading "/" given, pulls from given directory

  -r {zeros,previousgood,stats}
                        String. Required. Replacement method of flagged data
                        in output raw data file. Can be
                        "zeros","previousgood", or "stats"

  -v VEGAS_DIR          If inputting a VEGAS spectral line mode file, enter
                        AGBT19B_335 session number (1/2) and bank (C/D) ex
                        "1D".
  -newfile OUTPUT_BOOL  Copy the original data and output a replaced datafile.
                        Default True. Change to False to not write out a whole
                        new GUPPI file

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
my_dir = '/data/scratch/Spring2020/'
out_dir = my_dir 


#argparse parsing
parser = argparse.ArgumentParser(description="""function description""")

#input file
parser.add_argument('-i',dest='infile',type=str,required=True,help='String. Required. Name of input filename. Automatically pulls from standard data directory. If leading "/" given, pulls from given directory')






#replacement method
parser.add_argument('-r',dest='method',type=str,choices=['zeros','previousgood','stats'], required=True,default='zeros',help='String. Required. Replacement method of flagged data in output raw data file. Can be "zeros","previousgood", or "stats"')




#vegas spectral line file? needs new data directory and session
parser.add_argument('-v',dest='vegas_dir',type=str,default='0',help='If inputting a VEGAS spectral line mode file, enter AGBT19B_335 session number (1/2) and bank (C/D) ex "1D".')

#write out a whole new raw file or just get SK/accumulated spectra results
parser.add_argument('-newfile',dest='output_bool',type=bool,default=True,help='Copy the original data and output a replaced datafile. Default True. Change to False to not write out a whole new GUPPI file')



#Save raw data to npy files (storage intensive, unnecessary)
parser.add_argument('-npy',dest='rawdata',type=bool,default=False,help='Boolean. True to save raw data to npy files. This is storage intensive and unnecessary since blimpy. Default is False')


#parse input variables
args = parser.parse_args()
infile = args.infile
method = args.method
rawdata = args.rawdata


v_s = args.vegas_dir[0]
if v_s != '0':
	v_b = args.vegas_dir[1]
	in_dir = in_dir+'vegas/AGBT19B_335_0'+v_s+'/VEGAS/'+v_b+'/'
output_bool = args.output_bool




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


base = npy_dir+infile[len(in_dir):-4]

#filenames to save to
#'p' stands for polarization
#sk_filename = base+'_SK_m'+str(SK_ints)+'_'+method+'_s'+str(sigma)+'_'+rfi+'.npy'
#flags_filename = base+'_flags_m'+str(SK_ints)+'_'+method+'_s'+str(sigma)+'_'+rfi+'.npy'
#spect_filename = base+'_spect_m'+str(SK_ints)+'_'+method+'_s'+str(sigma)+'_'+rfi+'.npy'




if rawdata:
	print('Saving raw data to npy block style files')

#init copy of file for replaced data
print('Getting output datafile ready...')
outfile = out_dir + infile[len(in_dir):-4]+'_RFImitigated_'+infile[-4:]



#--------------------------------------
# Fun
#--------------------------------------


start_time = time.time()


flagged_pts_p1 = 0
flagged_pts_p2 = 0


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

	print('Data shape: {} || block size: {}'.format(data.shape,data.nbytes))

	#save raw data
	if rawdata:
		#pad number to three digits
		block_fname = str(block).zfill(3)

		save_fname = base+'_block'+block_fname+'.npy'
		np.save(save_fname,data)
		#print('Saved under '+out_dir+save_fname)





	#Calculations

	#ASSUMING NPOL = 2:

	


	#======================================
	#insert RFI detection of choice
	#you currently have:
	# data : 3D array of data from a single GUPPI/VPM block 'block'
	#	index [Channel,Spectrum,Polarization]
	#	these are np.complex64 complex channelized voltages
	#
	#you need to output from this section:
	# repl_chunk : 3D array of flags
	#	same shape as data
	#	0: unflagged || 1: flagged
	#
	#
	#
	#	Your RFI code goes here
	#
	#
	#======================================

	np.save('')


	#Replace data
	print('Calculations complete...')
	print('Replacing Data...')
	

	#now flag shape is (chan,spectra,pol)
	#repl_chunk[:,:,0][repl_chunk[:,:,1]==1]=1
	#repl_chunk[:,:,1][repl_chunk[:,:,0]==1]=1
	
	#record flagging % in both polarizations
	flagged_pts_p1 += (1./numblocks) * ((100.*np.count_nonzero(repl_chunk[:,:,0]))/repl_chunk[:,:,0].size)
	flagged_pts_p2 += (1./numblocks) * ((100.*np.count_nonzero(repl_chunk[:,:,1]))/repl_chunk[:,:,1].size)


	#now flag shape is (chan,spectra,pol)
	#apply union of flags between the pols
	repl_chunk[:,:,0][repl_chunk[:,:,1]==1]=1
	repl_chunk[:,:,1][repl_chunk[:,:,0]==1]=1

	

	if method == 'zeros':
		#replace data with zeros
		data = repl_zeros(data,repl_chunk)

	if method == 'previousgood':
		#replace data with previous (or next) good
		data = previous_good(data,repl_chunk,SK_ints)

	if method == 'stats':
		#replace data with statistical noise derived from good datapoints
		data = statistical_noise(data,repl_chunk)

	#Write back to block
	if output_bool:
		print('Re-formatting data and writing back to file...')
		data = guppi_format(data)
		out_rawFile.write(data.tostring())








print('Pol0: '+str(flagged_pts_p1)+' datapoints were flagged out of '+str(tot_points))


print('Pol1: '+str(flagged_pts_p2)+' datapoints were flagged out of '+str(tot_points))






print('Saved replaced data to '+outfile)


end_time = time.time()
elapsed = float(end_time-start_time)/60 

print('Program took {} minutes'.format(elapsed))

print('Done!')
