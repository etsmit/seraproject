#guppi_org_by_coarsechan.py
#reorganizes npy results from guppiraw.py by coarsechannel instead of blocknumber
#this is nice because then (FFTLEN * number of blocks) integer divides the amount of time samples
# in each block - so no dropped data

#-------------------------------------------

import numpy as np
import os,sys
import matplotlib.pyplot as plt

import scipy as sp
import scipy.optimize
import scipy.special

import commands
import time

#from SK_in_Python import *

#-------------------------------------------
#Inputs

my_dir = '/home/scratch/esmith/RFI_MIT/'

#input directory
if sys.argv[1][0] != '/':
	work_dir = my_dir + sys.argv[1]
else:
	work_dir = sys.argv[1]

#output directory (for now, is the same)
#output base filename
#outfile_base = sys.argv[2]

#number of coarse chans (can read from file, this is a little easier for now)
numchan = sys.argv[2]

#hardcode for GBT19A-479 for now
numchan=256
outfile_base = '58626_A002308_0025_0000_'


#----------------------------------------------
#Fun


blocks = commands.getoutput('ls '+work_dir).split('\n')


#print('Datablock files:')
#print(blocks)

print('Amount of files: '+str(len(blocks)))


for i in range(256):
	coarsechan=str(i)
	print('Coarse Channel: '+coarsechan)	
	outdata=np.zeros((128,1032704,2),dtype=np.complex64)#hardcode alert
	print('Loading files/Filling channel data...')
	for j in range(len(blocks)):
		print(str(j+1)+'/'+str(len(blocks)))
		indata=np.load(work_dir+blocks[j])
		print('Indata size: '+str(indata.nbytes/10e8)+' GB')
		outdata[i,:,:] = indata[i,:,:]
		indata = None
		print('Outdata size: '+str(outdata.nbytes/10e8)+' GB')
	print('Coarse channel filled. Reshaping...')
	outdata = np.array(outdata)
	outdata = np.reshape(outdata,(outdata.shape[0]*outdata.shape[1],2))
	print('New array size: '+str(outdata.shape))


	if i < 100:
		coarsechan = '0'+coarsechan
	if i < 10:
		coarsechan = '0'+coarsechan
			
	outfile = work_dir+outfile_base+'chan'+coarsechan+'.npy'
	print('Saving to '+outfile)
	np.save(outfile,outdata)
			
			
		
	









