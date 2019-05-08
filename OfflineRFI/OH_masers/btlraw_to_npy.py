#--------------------------------------------------
#btlraw_to_npy
#opens BTLraw data files (ex. GBT18B_342) and turns them into npy files)
# -- dependent on blimpy (UCBerkeleySETI/blimpy on github)
#Takes two inputs:
#1: input filename (BL raw format)
#2: npy file to save to !!(do NOT attach .npy on the end - i'll do that)!!
#--------------------------------------------------

import numpy as np
from blimpy import GuppiRaw
import os,sys

#input variables

#directory to pull from - hardcoding this becuase it should stay the same (for my purposes)
my_dir = '/home/scratch/esmith/RFI_MIT/'

inputFileName = my_dir+sys.argv[1]
outfile = my_dir+'npybtldata/blocks/'+sys.argv[2]

print('Opening file: '+inputFileName)
rawFile = GuppiRaw(inputFileName)

numblocks = rawFile.find_n_data_blocks()
print('File has '+str(numblocks)+' data blocks')



#Smaller raw BTL files are not filled to their byte max, and will have less blocks (I think)
for blockNumber in range(numblocks):
	print('---------------------')
	print('Block '+str(blockNumber+1)+' of '+str(numblocks))

	header,data = rawFile.read_next_data_block(blockNumber)
	if blockNumber == 0:
		print(header)
	
	save_fname = outfile+'block'+str(blockNumber)+'.npy'
	np.save(save_fname,data)
print('Done!')
	













