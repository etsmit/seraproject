#--------------------------------------------------
"""
detectRFI_VPM_dev.py


Main program for detecting/excising RFI in seraproj
dev version, in-progress and untested changes
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

from RFI_detection import *
from RFI_support import *

#--------------------------------------
# Inputs
#--------------------------------------

#in_dir = '/export/home/ptcs/scratch/raw_RFI_data/'#using maxwell (no longer available?)
#in_dir = '/lustre/pulsar/users/rlynch/RFI_Mitigation/'#using lustre (no longer available)
in_dir = '/data/rfimit/unmitigated/rawdata/'#leibniz only
out_dir = '/data/scratch/SKresults/'#leibniz only
jstor_dir = '/jetstor/scratch/SK_raw_data_results/'#leibniz only

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


num_iter = 0
failed = 0

#input file
#pulls from the raw data directory if full path not given
if infile[0] != '/':
	infile = in_dir + infile
else:
	in_dir = infile[:infile.rfind('/')+1]
	#infile = infile[infile.rfind('/')+1:]

if infile[-4:] != '.raw':
	print("WARNING input filename doesn't end in '.raw'. Are you sure you want to use this file?")

#--------------------------------------
# Inits
#--------------------------------------


npybase = out_dir+'npy_results/'+infile[len(in_dir):-4]

#filenames to save to
ms_sk_filename = f"{npybase}_MSSK_m{SK_M}_{method}_s{sigma}_{rfi}_ms{ms0}-{ms1}_{cust}.npy"
ms_spect_filename = f"{npybase}_MSspect_m{SK_M}_{method}_s{sigma}_{rfi}_ms{ms0}-{ms1}_{cust}.npy"
sk_filename = f"{npybase}_SK_m{SK_M}_{method}_s{sigma}_{rfi}_{cust}.npy"
flags_filename = f"{npybase}_flags_m{SK_M}_{method}_s{sigma}_{rfi}_ms{ms0}-{ms1}_{cust}.npy"
spect_filename = f"{npybase}_spect_m{SK_M}_{method}_s{sigma}_{rfi}_{cust}.npy"
regen_filename = f"{npybase}_regen_m{SK_M}_{method}_s{sigma}_{rfi}_mb{mb}_{cust}.npy"
#outfile = f"{out_dir+'raw_results/'}{infile[len(in_dir):-4]}_{method}_m{SK_M}_s{sigma}_{rfi}_ms{ms0}-{ms1}_mb{mb}_{cust}{infile[-4:]}"
outfile = f"{jstor_dir}{infile[len(in_dir):-4]}_{method}_m{SK_M}_s{sigma}_{rfi}_ms{ms0}-{ms1}_mb{mb}_{cust}{infile[-4:]}"


#threshold calc from sigma
SK_p = (1-scipy.special.erf(sigma/math.sqrt(2))) / 2
print('Probability of false alarm: {}'.format(SK_p))

#calculate thresholds
print('Calculating SK thresholds...')
lt, ut = SK_thresholds(SK_M, N = n, d = d, p = SK_p)
print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))

#mslt,msut =  SK_thresholds(SK_M-(ms1-1), N = n, d = d, p = SK_p)

#print


if rawdata:
	print('Saving raw data to npy block style files')


#--------------------------------------
# Fun
#--------------------------------------


start_time = time.time()


#os.system('rm '+outfile)
if output_bool:
	print('Saving replaced data to '+outfile)
	#print(infile,outfile)
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
	print(infile)
	#print header for the first block
	if block == 0:
		header,headersize = rawFile.read_header()
		print('Header size: {} bytes'.format(headersize))

	#loading multiple blocks at once?	
	for mb_i in range(mb):
		if mb_i==0:
			header,data = rawFile.read_next_data_block()
			data = np.copy(data)
			#length in spectra of one block, for use during rewriting mit. data
			d1s = data.shape[1]
		else:
			h2,d2 = rawFile.read_next_data_block()
			data = np.append(data,np.copy(d2),axis=1)

	#data is channelized voltages

	#print header for the first block
	if block == 0:
		print('Datatype: '+str(type(data[0,0,0])))
		for line in header:
			print(line+':  '+str(header[line]))


	#find data shape
	num_coarsechan = data.shape[0]
	num_timesamples= data.shape[1]
	num_pol = data.shape[2]
	print('Data shape: {} || block size: {}'.format(data.shape,data.nbytes))

	#save raw data?
	if rawdata:
		#pad number to three digits
		block_fname = str(block).zfill(3)
		save_fname = npybase+'_block'+block_fname+'.npy'
		np.save(save_fname,data)
		#print('Saved under '+out_dir+save_fname)



	#Check to see if SK_M divides the total amount of data points
	mismatch = num_timesamples % SK_M
	if (mismatch != 0):
		print('Warning: SK_M does not divide the amount of time samples')
		#exit()

	
	print('There are {} time samples and you input {} as m'.format(num_timesamples,SK_M))
	num_SKbins = int(num_timesamples/SK_M)
	print('Leading to '+str(num_SKbins)+' SK time bins')


	#Calculations

	#ASSUMING NPOL = 2:
	#init s1,s2,ms_s1,ms_s2
	s1 = np.zeros((num_coarsechan,num_SKbins,2))
	s2 = np.zeros((num_coarsechan,num_SKbins,2))

	ms_s1 = np.zeros((num_coarsechan-(ms0-1),num_SKbins-(ms1-1),2))
	ms_s2 = np.zeros((num_coarsechan-(ms0-1),num_SKbins-(ms1-1),2))

	ms_data = np.zeros((num_coarsechan-(ms0-1),num_time_samples-(ms1-1),2))

	#make multiscale data
	for ichan in range(ms0):
		for itime in range(ms1):
			ms_data += (data[ichan:ichan+(num_coarsechan-(ms0-1)),itime:itime+(num_SKbins-(ms1-1)),:])


	#make s1 and s2 arrays, as well as avg spects
	for k in range(num_SKbins):
	
		#take the correct time selection of data
		start = k*SK_M
		end = (k+1)*SK_M
		data_chunk = data[:,start:end,:]
		ms_data_chunk = ms_data[ ]

		data_chunk = np.abs(data_chunk)**2
		ms_data_chunk = np.abs(ms_data_chunk)**2

		s1[:,k,:] = np.sum(data_chunk,axis=1)
		s2[:,k,:] = np.sum(data_chunk**2,axis=1)


		#the last time bin of the multiscale window will have slightly shorter length
		if (k+1 != num_SKbins):
			ms_data_chunk = ms_data[:,start:end,:]
		else:
			ms_data_chunk = ms_data[:,start:end-(ms1-1),:]

		ms_data_chunk = np.abs(ms_data_chunk)**2

		ms_s1[:,k,:] = np.sum(ms_data_chunk,axis=1)
		ms_s2[:,k,:] = np.sum(ms_data_chunk**2,axis=1)

		#square it
		spectrum = np.average(data_chunk,axis=1)


		if (k==0):
			spect_block=np.expand_dims(spectrum,axis=2)
		else:
			spect_block=np.c_[spect_block,np.expand_dims(spectrum,axis=2)]

	#do singlescale SK flagging
	sk_block = SK_EST_alt(s1,s2,SK_M,n=1,d=1)
	flags_block = np.zeros(sk_block.shape,dtype=np.int8)
	flags_block[sk_block>ut] = 1
	flags_block[sk_block<lt] = 1

	#==================================================================

	#perform multiscale SK
	for k in range(num_SKbins-(ms1-1)):
	

		sk_spect = np.zeros((num_coarsechan-(ms0-1),2))

		#perform RFI detection
		if (rfi == 'SKurtosis'):
			#print(ms_s1.shape)
			#print(ms_s2.shape)
			sk_spect[:,0] = ms_SK_EST(ms_s1[:,k,0],ms_s2[:,k,0],SK_M-(ms1-1),n,d)
			sk_spect[:,1] = ms_SK_EST(ms_s1[:,k,1],ms_s2[:,k,1],SK_M-(ms1-1),n,d)
			#init flag chunk
			ms_flag_spect = np.zeros((num_coarsechan-(ms0-1),2),dtype=np.int8)
			#flag (each pol separately, for records)
			ms_flag_spect[sk_spect>ut] = 1
			ms_flag_spect[sk_spect<lt] = 1


		elif (rfi == 'SEntropy'):
			#not done?
			sk_spect[:,0] = entropy(data_chunk[:,:,0])
			sk_spect[:,1] = entropy(data_chunk[:,:,1])
			#init flag chunk
			flag_spect = np.zeros((num_coarsechan,2),dtype=np.int8)
			#flag (each pol separately, for records)
			#flag_spect[sk_spect>ut] = 1
			#flag_spect[sk_spect<lt] = 1


		
		#append to results
		if (k==0):
			ms_sk_block=np.expand_dims(sk_spect,axis=2)
			ms_flags_block = np.expand_dims(ms_flag_spect,axis=2)

		else:
			ms_sk_block=np.c_[ms_sk_block,np.expand_dims(sk_spect,axis=2)]
			ms_flags_block = np.c_[ms_flags_block,np.expand_dims(ms_flag_spect,axis=2)]


	#adj_chan flagging here
	#flags_block = adj_chan_skflags(spect_block,flags_block,sk_block,1,3)
	
	ms_flags_block = np.transpose(ms_flags_block,(0,2,1))

	#print flagged percentage for both pols from single scale SK
	print('{}/{}% flagged'.format((100.*np.count_nonzero(flags_block[:,:,0])/flags_block[:,:,1].size),(100.*np.count_nonzero(flags_block[:,:,1])/flags_block[:,:,1].size)))

	#print flagged percentage for both pols from multi scale SK
	print('{}/{}% flagged'.format((100.*np.count_nonzero(ms_flags_block[:,:,0])/ms_flags_block[:,:,1].size),(100.*np.count_nonzero(ms_flags_block[:,:,1])/ms_flags_block[:,:,1].size)))


	#apply union of single scale and multiscale flag masks
	#(each ms flag pixel covers several single scale pixels)
	#print(flags_block.shape,ms_flags_block.shape)
	for ichan in range(ms0):
		for itime in range(ms1):
			#print(flags_block.shape)
			#print(ms_flags_block.shape)
			flags_block[ichan:ichan+(num_coarsechan-(ms0-1)),itime:itime+(num_SKbins-(ms1-1)),:][ms_flags_block==1] = 1


	#print flagged percentage for both pols from union of ss and ms SK
	print('{}/{}% flagged'.format((100.*np.count_nonzero(flags_block[:,:,0])/flags_block[:,:,1].size),(100.*np.count_nonzero(flags_block[:,:,1])/flags_block[:,:,1].size)))

	print(flags_block.shape,ms_flags_block.shape)

	if (block==0):
		sk_all = sk_block
		ms_sk_all = ms_sk_block
		spect_all = spect_block
		ms_spect_all = ms_s1
		flags_all = flags_block
	else:
		sk_all = np.concatenate((sk_all,sk_block),axis=1)
		ms_sk_all = np.c_[ms_sk_all,ms_sk_block]
		spect_all = np.c_[spect_all,spect_block]
		ms_spect_all = np.concatenate((ms_spect_all,ms_s1),axis=1)
		flags_all = np.concatenate((flags_all,flags_block),axis=1)

	#print('shapecheck')
	#print(f'blocks: sk {sk_block.shape} spect {spect_block.shape} f {flags_block.shape}')
	#print(f'all: sk {sk_all.shape} spect {spect_all.shape} f {flags_all.shape}')

	#Replace data
	print('Calculations complete...')
	print('Replacing Data...')
	
	repl_chunk=np.copy(flags_block)
	#print(repl_chunk.shape)
	print('transposed')
	#now flag shape is (chan,spectra,pol)
	#apply union of flags across pols
	repl_chunk[:,:,0][repl_chunk[:,:,1]==1]=1
	repl_chunk[:,:,1][repl_chunk[:,:,0]==1]=1
	print('union')

	
	#extend flagging array to be same size as raw data
	extend = np.ones((1,SK_M,1),dtype=np.int8)
	print('extend')
	repl_chunk = np.kron(repl_chunk,extend).astype(np.int8)
	print('repl_chunk set...')
	#sir flagging?
	#for i in range(2):
	#	repl_chunk[:,:,0] = sir(repl_chunk[:,:,0],0.2,0.2,'union')
	#	repl_chunk[:,:,1] = sir(repl_chunk[:,:,1],0.2,0.2,'union')

	if method == 'zeros':
		#replace data with zeros
		data = repl_zeros(data,repl_chunk)

	if method == 'previousgood':
		#replace data with previous (or next) good
		data = prevgood_init(data,repl_chunk,SK_M)
		#data = previous_good(data,repl_chunk,SK_M)

	if method == 'stats':
		print(num_iter,failed)
		#replace data with statistical noise derived from good datapoints
		#data = statistical_noise_alt_fir(data,repl_chunk,SK_M)
		data = statistical_noise_fir_abs(data,repl_chunk,'/data/scratch/SKresults/absorber_rms/vegas_59934_72163_ABSORBER_0024.0000_rms.npy',block)
		flag_track = np.c_[np.ones(data.shape[0])*data.shape[1],np.count_nonzero(repl_chunk[:,:,0],axis=1)]
		np.save(f'/data/scratch/SKresults/absorber_rms/flag_track{block}.npy',flag_track)
		#num_iter += thisnum_iter
		#failed += this_failed


	#generate averaged datafile from replaced data
	for k in range(num_SKbins):
	
		#take the stream of correct data
		start = k*SK_M
		end = (k+1)*SK_M
		data_chunk = data[:,start:end,:]


		data_chunk = np.abs(data_chunk)**2#abs value and square

		regen = np.average(data_chunk,axis=1)

		if (k==0):
			regen_block=np.expand_dims(regen,axis=2)
		else:
			regen_block=np.c_[regen_block,np.expand_dims(regen,axis=2)]

	if (block==0):
		regen_all = regen_block
	else:
		regen_all = np.c_[regen_all,regen_block]



	#Write back to copied raw file
	if output_bool:
		print('Re-formatting data and writing back to file...')
		for mb_i in range(mb):
			out_rawFile.seek(headersize,1)
			d1 = guppi_format(data[:,d1s*mb_i:d1s*(mb_i+1),:])
			out_rawFile.write(d1.tostring())
		#out_rawFile.seek(headersize,1)
		#d2 = guppi_format(data[:,d1s:,:])
		#out_rawFile.write(d1.tostring())


#save SK results


np.save(sk_filename, sk_all)
print(f'{sk_all.shape} SK spectra saved in {sk_filename}')


#save ms_SK results
ms_sk_all = np.transpose(ms_sk_all,(0,2,1))

np.save(ms_sk_filename, ms_sk_all)
print(f'{ms_sk_all.shape} ms_SK spectra saved in {ms_sk_filename}')

np.save(ms_spect_filename,ms_spect_all)
print(f'{ms_spect_all.shape} ms_spect spectra saved in {ms_spect_filename}')



#save spectrum results
spect_all = np.transpose(spect_all,(0,2,1))
np.save(spect_filename, spect_all)
print(f'{spect_all.shape} Unmitigated spectra saved in {spect_filename}')


#save mitigated results results
regen_all = np.transpose(regen_all,(0,2,1))
np.save(regen_filename, regen_all)
print(f'{regen_all.shape} Mitigated spectra saved in {regen_filename}')


#save flags results
#flags_all = np.transpose(flags_all,(0,2,1))
np.save(flags_filename,flags_all)
print(f'{flags_all.shape} Flags file saved to {flags_filename}')

print(f"'{spect_filename}','{flags_filename}','{regen_filename}','{sk_filename}'")
#put files in log file for easy inspection later
logf = '/data/scratch/SKresults/logf.txt'
os.system(f"""echo "'{spect_filename}','{flags_filename}','{regen_filename}','{sk_filename}'" >> {logf}""")


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
f_union = np.copy(flags_all)
f_union[:,:,0][f_union[:,:,1]==1]=1
f_union[:,:,1][f_union[:,:,0]==1]=1
flagged_union_pts = (100.*np.count_nonzero(f_union))/f_union.size

print('Union of flags: {}% of data flagged'.format(flagged_union_pts))

#print(f'Percentage of failed good data replacement: {100.*failed/num_iter}%')


print('Saved replaced data to '+outfile)


end_time = time.time()
elapsed = float(end_time-start_time)/60 

print('Program took {} minutes'.format(elapsed))

print('Done!')
