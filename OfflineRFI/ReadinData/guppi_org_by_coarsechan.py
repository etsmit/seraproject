#guppi_org_by_coarsechan.py
#reorganizes npy results from guppiraw.py by coarsechannel instead of blocknumber
#this is nice because then (FFTLEN * number of blocks) integer divides the amount of time samples
# in each block - so no dropped data

#uses 8 cores and roughly 50 GB of memory

#-------------------------------------------

import numpy as np
import os,sys
import matplotlib.pyplot as plt

import scipy as sp
import scipy.optimize
import scipy.special

import commands
import time

import multiprocessing
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
#Functions

def filler(i,blocks):
	#coarse channel i
	coarsechan=str(i)
	print('Coarse Channel: '+coarsechan)	
	outdata=np.zeros((128,1032704,2),dtype=np.complex64)#hardcode alert
	for j in range(len(blocks)):
		print(str(j+1)+'/'+str(len(blocks)))
		indata=np.load(work_dir+blocks[j])
		outdata[j,:,:] = indata[i,:,:]
		indata = None
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



#----------------------------------------------
#Fun

start = time.time()

blocks = commands.getoutput('ls '+work_dir).split('\n')
blocks = [block for block in blocks if block[24]=='b']


#print('Datablock files:')
#print(blocks)

print('Amount of files: '+str(len(blocks)))



for chan1 in range(8,256,8):
	print('Channels '+str(chan1)+' - '+str(chan1+7))
	p1 = multiprocessing.Process(target=filler,args=(chan1+0,blocks,))
	p2 = multiprocessing.Process(target=filler,args=(chan1+1,blocks,))
	p3 = multiprocessing.Process(target=filler,args=(chan1+2,blocks,))
	p4 = multiprocessing.Process(target=filler,args=(chan1+3,blocks,))
	p5 = multiprocessing.Process(target=filler,args=(chan1+4,blocks,))
	p6 = multiprocessing.Process(target=filler,args=(chan1+5,blocks,))
	p7 = multiprocessing.Process(target=filler,args=(chan1+6,blocks,))
	p8 = multiprocessing.Process(target=filler,args=(chan1+7,blocks,))
	print('Starting 8 processes')
	p1.start()
	p2.start()
	p3.start()
	p4.start()
	p5.start()
	p6.start()
	p7.start()
	p8.start()
	print('Waiting to fill...')
	p1.join()
	p2.join()
	p3.join()
	p4.join()
	p5.join()
	p6.join()
	p7.join()
	p8.join()
	#mid = time.time()
	print('Time so far:')
	print(mid-start)

stop = time.time()
print('Total time: '+str(stop-start))

print('Done')







