#plots baselines


#Imports
import numpy as np
import os,sys
import matplotlib.pyplot as plt

import scipy as sp
import scipy.optimize
import scipy.special
import math as math

import time

from blimpy import GuppiRaw

from SK_in_Python import *



#INPUTS



infile_orig = sys.argv[1]

infile_excised = sys.argv[2]

SK_ints = int(sys.argv[3])




#--------------------------------------
# Inits
#--------------------------------------



#filenames to save to
#'p' stands for polarization
#spect_npy_p1 = 'J0332_spect_m'+str(SK_ints)+'_p1.npy'
#spect_npy_p2 = 'J0332_spect_m'+str(SK_ints)+'_p2.npy'



#array to hold spectrum results
spect_orig_p1 = []
spect_orig_p2 = []
spect_excised_p1 = []
spect_excised_p2 = []






#--------------------------------------
# Fun
#--------------------------------------


start_time = time.time()




#load file and copy
print('Opening file: '+infile_orig)
rawFile_orig = GuppiRaw(infile_orig)
print('Opening file: '+infile_excised)
rawFile_yeet = GuppiRaw(infile_excised)



numblocks = rawFile_orig.find_n_data_blocks()
print('File has '+str(numblocks)+' data blocks')



for block in range(numblocks):
	print('#--------------------------------------')
	print('Block: '+str(block))
	if block == 0:
		header,headersize = rawFile_orig.read_header()
		print('Header size: {} bytes'.format(headersize))
	header,data = rawFile_orig.read_next_data_block()
	header,yeet = rawFile_yeet.read_next_data_block()

	
	#print header for the first block
	if block == 0:
		print('Datatype: '+str(type(data[0,0,0])))
		for line in header:
			print(line+':  '+str(header[line]))



	num_coarsechan = data.shape[0]
	num_timesamples= data.shape[1]
	# ^^^ FYI these are time samples of voltage corresponding to a certain frequency
	# See the notebook drawing on pg 23
	# FFT has already happened in the roaches
	num_pol = data.shape[2]

	print('Data shape: '+str(data.shape))


	#Check to see if SK_ints divides the total amount of data points
	mismatch = num_timesamples % SK_ints
	kept_samples = int(num_timesamples- mismatch)
	SK_timebins = int(kept_samples/SK_ints)


	if mismatch != 0:
		data = data[:,:kept_samples,:]



	#Calculations
	for j in range(num_pol):
		flagged_pts=0
		#print('Polarization '+str(j))
		for k in range(SK_timebins):
	
			#take the stream of correct data
			start = k*SK_ints
			end = (k+1)*SK_ints
			data_chunk = data[:,start:end,j]
			yeeted_chunk = yeet[:,start:end,j]

			#square it
			data_chunk = np.abs(data_chunk)**2#abs value and square
			yeeted_chunk = np.abs(yeeted_chunk)**2#abs value and square

			spectrum = np.average(data_chunk,axis=1)
			excised_spectrum = np.average(yeeted_chunk,axis=1)

		

			#append to results
			if j:
				spect_orig_p2.append(spectrum)
				spect_excised_p2.append(spectrum)

			else:
				spect_orig_p1.append(spectrum)
				spect_excised_p1.append(spectrum)



	


orig_p1 = np.average(spect_orig_p1,axis=1)
excised_p1 = np.average(spect_excised_p1,axis=1)
orig_p2 = np.average(spect_orig_p2,axis=1)
excised_p2 = np.average(spect_excised_p2,axis=1)

import matplotlib.pyplot as plt

plt.plot(orig_p1,'r-',label='Original XX')
plt.plot(excised_p1,'r',linestyle=(0,(1,1)),label='Excised XX')
plt.plot(orig_p2,'b-',label='Original YY')
plt.plot(excised_p2,'b',linestyle=(0,(1,1)),label='Excised XX')

plt.show()
















