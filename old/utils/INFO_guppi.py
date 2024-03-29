#INFO_btl.py
#simply reads the header of a BTL/GUPPI raw file data block for basic file info


import sys,os
from blimpy import GuppiRaw
import numpy as np

my_dir = '/home/scratch/esmith/RFI_MIT/'


#pulls from my scratch directory if full path not given
if sys.argv[1][0] != '/':
	inputFileName = my_dir + sys.argv[1]
else:
	inputFileName = sys.argv[1]


print('Reading in GUPPI raw file...')
r = GuppiRaw(inputFileName)

print('Parsing Header...')
hdr,data = r.read_header()


print('----------------------------------')
for line in hdr:
	print(line+':  '+str(hdr[line]))
print('----------------------------------')

print('Number of blocks: '+str(r.find_n_data_blocks()))
print('Reading data shape...')
print('This may take a while, feel free to kill the process')

#hdr,data = r.read_next_data_block(0)

#print('Data Shape: '+str(data.shape))

