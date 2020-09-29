#routine to reduce two ON/OFF guppiraw files into a single spectrum
#try to mimic gbtidl's getps as close as possible

#usage:
#python guppiraw_spect.py ON_file_1 ON_file_2 OFF_file_1 OFF_file_2



import sys,os
import numpy as np
import matplotlib.pyplot as plt
import time

from blimpy import GuppiRaw


out_dir = '/home/scratch/esmith/'
out_filename = out_dir + 'test_ps.npy'

in_dir = '/data/rfimit/unmitigated/rawdata/'

#ON scan
ON_filename_1 = sys.argv[1]
print('Loading ON scan...')
ON_file = GuppiRaw(in_dir+ON_filename_1)

#OFF scan
OFF_filename_1 = sys.argv[3]
print('Loading OFF scan...')
OFF_file = GuppiRaw(in_dir+OFF_filename_1)


out_dir = '/home/scratch/esmith/'
out_filename = out_dir + 'test_ps.npy'


numblocks = ON_file.find_n_data_blocks()
print('Files have '+str(numblocks)+' data blocks')





for block in range(numblocks):
	print('------------------------------------------')
	print('Block: '+str(block))
	if block == 0:
		header,headersize = ON_file.read_header()
		print('Header size: {} bytes'.format(headersize))
	
	ON_h, ON_block = ON_file.read_next_data_block()
	OFF_h, OFF_block = OFF_file.read_next_data_block()

	#protect against divide by 0
	OFF_block[OFF_block==0]=1e-3
	

	Tsys = 1.00#TODO: how to figure this out?
	
	#compute whole block of position-switched data
	ps_block = Tsys*((ON_block - OFF_block) / (OFF_block))
	
	spect = np.average(ps_block,axis=1)
	spect = np.expand_dims(spect,axis=2)

	if (block==0):
		ps_spect = spect
	else:
		ps_spect=  np.c_[ps_spect,spect]

#========================================
#from second set of files


#ON scan
ON_filename_2 = sys.argv[2]
print('Loading ON scan...')
ON_file = GuppiRaw(in_dir+ON_filename_2)

#OFF scan
OFF_filename_2 = sys.argv[4]
print('Loading OFF scan...')
OFF_file = GuppiRaw(in_dir+OFF_filename_2)

for block in range(numblocks):
	print('------------------------------------------')
	print('Block: '+str(block))
	if block == 0:
		header,headersize = ON_file.read_header()
		print('Header size: {} bytes'.format(headersize))
	
	ON_h, ON_block = ON_file.read_next_data_block()
	OFF_h, OFF_block = OFF_file.read_next_data_block()

	#protect against divide by 0
	OFF_block[OFF_block==0]=1e-3
	

	Tsys = 1.00#TODO: how to figure this out?
	
	#compute whole block of position-switched data
	ps_block = Tsys*((ON_block - OFF_block) / (OFF_block))
	
	spect = np.average(ps_block,axis=1)
	spect = np.expand_dims(spect,axis=2)

	ps_spect=  np.c_[ps_spect,spect]


out_spect = np.average(ps_spect,axis=2)

np.save(out_filename,out_spect)

s2 = np.abs(out_spect)**2

plt.plot(s2[:,0],'r',linewidth=0.5,label='Pol0')
plt.plot(s2[:,1],'b',linewidth=0.5,label='Pol1')
plt.xlim((0,128))
plt.legend()
plt.show()







