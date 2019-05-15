#--------------------------------------------------
#btlraw.py
#opens GUPPI data files and turns them into npy files)
# -- dependent on blimpy (UCBerkeleySETI/blimpy on github)
#Takes two inputs:
#1: input filename (BL/GUPPI raw format)
#2: directory to save npy files to (no '/' on the end)
#--------------------------------------------------

#--------------------------------------------------
# - FIRST - 
# Need to slightly modify blimpy code
# Simply copy OfflineRFI/ReadinData/guppi.py into your (pythonpath)/site-packages/blimpy/
# and overwrite the original.
# This modification allows for opening a given data block, rather than just the first one
#--------------------------------------------------


import numpy as np
from blimpy import GuppiRaw
import os,sys

#-----------------------------
# Input variables
#-----------------------------
my_dir = '/home/scratch/esmith/RFI_MIT/'

#pulls from my scratch directory if full path not given
if sys.argv[1][0] != '/':
	inputFileName = my_dir + sys.argv[1]
else:
	inputFileName = sys.argv[1]


outdir = my_dir+'/'+sys.argv[2]+'/'
os.system('mkdir '+outdir)

#-----------------------------
# Science
#-----------------------------

print('Opening file: '+inputFileName)
rawFile = GuppiRaw(inputFileName)

numblocks = rawFile.find_n_data_blocks()
print('File has '+str(numblocks)+' data blocks')


for blockNumber in range(numblocks):
	print('---------------------')
	print('Block '+str(blockNumber+1)+' of '+str(numblocks))

	header,data = rawFile.read_next_data_block(blockNumber)
	if blockNumber == 0:
		for line in header:
			print(line+':  '+str(header[line]))

	save_fname = sys.argv[2]+'_block'+str(blockNumber)+'.npy'
	np.save(outdir+save_fname,data)
	print('Saved under '+outdir+save_fname)
print('Done!')
	












