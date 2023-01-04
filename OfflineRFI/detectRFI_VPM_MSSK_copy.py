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

 - psrenv preffered, or
 - python3.6.5
 - use /users/esmith/.conda/envs/py365 conda environment on green bank machine
 - type ' -h' to see help message

Inputs
------------
  -h, --help            show this help message and exit
  -i INFILE             String. Required. Name of input filename.
                        Automatically pulls from standard data directory. If
                        leading "/" given, pulls from given directory
  -rfi {SKurtosis,SEntropy,IQRM}
                        String. Required. RFI detection method desired.
  -m SK_M               Integer. Required. "M" in the SK equation. Number of
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
  -ms multiscale SK     String. Multiscale SK bin size. 
                        2 ints : Channel size / Time size, ex '-ms 42' Default '11'
  -mb mb		For loading multiple blocks at once. Helps with finding good
                        data for replacing flagged data, but can balloon RAM usage. 
                        Default 1.

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

#from RFI_detection import *
#from RFI_support import *

#--------------------------------------
# Inputs
#--------------------------------------

#in_dir = '/export/home/ptcs/scratch/raw_RFI_data/'#using maxwell (no longer available?)
#in_dir = '/lustre/pulsar/users/rlynch/RFI_Mitigation/'#using lustre (no longer available)
in_dir = '/data/rfimit/unmitigated/rawdata/'#leibniz
out_dir = '/data/scratch/Summer2022/'#leibniz


#argparse parsing
parser = argparse.ArgumentParser(description="""function description""")

#input file
parser.add_argument('-i',dest='infile',type=str,required=True,help='String. Required. Name of input filename. Automatically pulls from standard data directory. If leading "/" given, pulls from given directory')

#RFI detection method
parser.add_argument('-rfi',dest='RFI',type=str,required=True,choices=['SKurtosis','SEntropy'],default='SKurtosis',help='String. Required. RFI detection method desired.')

#SK integrations. 'M' in the SK equation. Number of data points to perform SK on at once/average together for spectrogram. FYI 1032704 (length of each block) has prime divisors (2**9) and 2017.
parser.add_argument('-m',dest='SK_M',type=int,required=True,default=512,help='Integer. Required. "M" in the SK equation. Number of data points to perform SK on at once/average together for spectrogram. ex. 1032704 (length of each block) has prime divisors (2**9) and 2017. Default 512.')


#replacement method
parser.add_argument('-r',dest='method',type=str,choices=['zeros','previousgood','stats'], required=True,default='zeros',help='String. Required. Replacement method of flagged data in output raw data file. Can be "zeros","previousgood", or "stats"')



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

#multiscale bin shape.
parser.add_argument('-ms',dest='ms',type=str,default='1,1',help='Multiscale SK. 2 ints : ChanSpec. Put a comma between. Default "1,1"')

#custom filename tag (for adding info not already covered in lines 187
parser.add_argument('-cust',dest='cust',type=str,default='',help='custom tag to add to end of filename')

#using multiple blocks at once to help stats replacement
parser.add_argument('-mult',dest='mb',type=int,default=1,help='load multiple blocks at once to help with stats/prevgood replacement')



#parse input variables
args = parser.parse_args()
infile = args.infile
SK_M = args.SK_M
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
rfi = args.RFI
ms = (args.ms).split(',')
ms0 = int(ms[0])
ms1 = int(ms[1])
cust = args.cust
mb = args.mb


#input file
#pulls from my scratch directory if full path not given
if infile[0] != '/':
	infile = in_dir + infile
else:
	in_dir = infile[:infile.index('/')+1]

if infile[-4:] != '.raw':
	print("WARNING input filename doesn't end in '.raw'. Are you sure you want to use this file?")

#--------------------------------------
# Inits
#--------------------------------------


base = out_dir+infile[len(in_dir):-4]

#filenames to save to
ms_sk_filename = f"{base}_SK_m{SK_M}_{method}_s{sigma}_{rfi}_ms{ms0}-{ms1}_{cust}.npy"
ms_spect_filename = f"{base}_spect_m{SK_M}_{method}_s{sigma}_{rfi}_ms{ms0}-{ms1}_{cust}.npy"
sk_filename = f"{base}_SK_m{SK_M}_{method}_s{sigma}_{rfi}_{cust}.npy"
flags_filename = f"{base}_flags_m{SK_M}_{method}_s{sigma}_{rfi}_ms{ms0}-{ms1}_{cust}.npy"
spect_filename = f"{base}_spect_m{SK_M}_{method}_s{sigma}_{rfi}_{cust}.npy"
regen_filename = f"{base}_regen_m{SK_M}_{method}_s{sigma}_{rfi}_mb{mb}_{cust}.npy"
outfile = f"{out_dir}{infile[len(in_dir):-4]}_{method}_m{SK_M}_s{sigma}_{rfi}_ms{ms0}-{ms1}_mb{mb}_{cust}{infile[-4:]}"




if rawdata:
	print('Saving raw data to npy block style files')


#flatten data array 'a' into format writeable to guppi file
#@jit(nopython=True)
def guppi_format(a):
	#takes array of np.complex64,ravels it and outputs as 1D array of signed
	#8 bit integers ordered real,imag,real,imag,.....
	#init output
	out_arr = np.empty(shape=2*a.size,dtype=np.int8)
	#get real values, ravel, cast to int8
	#a_real = a.ravel().real.astype(np.int8)
	arav = a.ravel()
	a_real = np.clip(np.floor(arav.real),-128,127).astype(np.int8)
	#get imag values, ravel, cast to int8
	#a_imag = a.ravel().imag.astype(np.int8)
	a_imag = np.clip(np.floor(arav.imag),-128,127).astype(np.int8)
	#interleave
	out_arr[::2] = a_real
	out_arr[1::2] = a_imag
	return out_arr




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
#check for mismatched amount of blocks
mismatch = numblocks % mb
if (mismatch != 0):
	print(f'There are {numblocks} blocks and you set -mb {mb}, pick a divisible integer')
	#exit()


for block in range(numblocks//mb):
	print('------------------------------------------')
	print(f'Block: {(block*mb)+1}/{numblocks}')
	#print header for the first block
	if block == 0:
		header,headersize = rawFile.read_header()
		print('Header size: {} bytes'.format(headersize))
		#print('Datatype: '+str(type(data[0,0,0])))
		for line in header:
			print(line+':  '+str(header[line]))

	#loading multiple blocks at once?
	#have to add them to a list of blocks because blimpy returns only a view of the block
	#so, when loading multiple blocks at the same time,
	#every block except the first is a copy of the last block
	#e.g. for mb=4, data ends up as block1,block4,block4,block4
	#creating the fake pulse!
	for mb_i in range(mb):
		if mb_i==0:
			header,data = rawFile.read_next_data_block()
			#length in spectra of one block, for use during rewriting mit. data
			d1s = data.shape[1]
			data = np.copy(data)
		else:
			h2,d2 = rawFile.read_next_data_block()
			data = np.append(data,np.copy(d2),axis=1)

	#data = np.append(data,d2,axis=1)



	data = np.ascontiguousarray(data)
	if block ==0:
		np.save('/home/scratch/esmith/testblock.npy',data)


	#data is channelized voltages



	#find data shape
	num_coarsechan = data.shape[0]
	num_timesamples= data.shape[1]
	num_pol = data.shape[2]
	print('Data shape: {} || block size: {}'.format(data.shape,data.nbytes))

	
	#out_test = np.arange(num_coarsechan*num_timesamples*num_pol,dtype=np.complex64) + 1.j*np.arange(num_coarsechan*num_timesamples*num_pol,dtype=np.complex64)
	#out_test = np.reshape(out_test,(num_coarsechan,num_timesamples,num_pol))
	if output_bool:
		print('Re-formatting data and writing back to file...')
		for mb_i in range(mb):
			out_rawFile.seek(headersize,1)
			#print(d1s*mb_i,d1s*(mb_i+1))
			#print(data[:,d1s*mb_i:d1s*(mb_i+1),:].shape)
			d1 = guppi_format(data[:,d1s*mb_i:d1s*(mb_i+1),:])
			#d1 = guppi_format(out_test[:,d1s*mb_i:d1s*(mb_i+1),:])
			#out_rawfile.write(headers[i])
			out_rawFile.write(d1.tostring())
		#out_rawFile.seek(headersize,1)
		#d2 = guppi_format(data[:,d1s:,:])
		#out_rawFile.write(d1.tostring())




print('Saved replaced data to '+outfile)


end_time = time.time()
elapsed = float(end_time-start_time)/60 

print('Program took {} minutes'.format(elapsed))

print('Done!')
