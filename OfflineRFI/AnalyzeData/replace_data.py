#replacing flagged data in a guppi file

print("""Accepted methods (input 3) are 'zeros', 'previousgood', 'stats'""") 


import sys,os
import numpy as np



from SK_in_Python import *

# I/O

my_dir = '/home/scratch/esmith/RFI_MIT/'

#pulls from my scratch directory if full path not given
if sys.argv[1][0] != '/':
	inputData = my_dir + sys.argv[1]
else:
	inputData = sys.argv[1]

if sys.argv[2][0] != '/':
	inputFlags = my_dir + sys.argv[2]
else:
	inputFlags = sys.argv[2]

method = sys.argv[3]

print('Loading npy files...')
data = np.load(inputData)
lil_flags = np.load(inputFlags)



print('Data shape: '+str(data.shape))
print('Flags shape: '+str(lil_flags.shape))


print('Exploding up flags file...')
big_flags,x = overlayflags_3D(data,lil_flags,180)
print(big_flags.shape)

bigflagsfile = '/home/scratch/esmith/RFI_MIT/bigflags.npy'
print('Saving bigflags file')
np.save(bigflagsfile,big_flags)

#--------------------------------------------------------------
#replace flagged portions of the data

if method == 'zeros':
	#replace data with zeros
	new_data = zeros(data,big_flags)

if method == 'previousgood':
	#replace data with previous (or next) good
	new_data = previous_good(data,big_flags,x)

if method == 'stats':
	#replace data with statistical noise derived from good datapoints
	new_data = statistical_noise(data,big_flags,x)

print('Done replacing')
outfile = '/home/scratch/esmith/RFI_MIT/replaced_data.npy'
print('Saving to '+outfile)
np.save(outfile,new_data)




