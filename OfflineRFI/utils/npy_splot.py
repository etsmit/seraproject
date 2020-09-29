#generates baseline spectra for comparison
#from npy files


import numpy as np
import matplotlib.pyplot as plt
import sys



my_dir = '/home/scratch/esmith/RFI_MIT/testing/'

#input spect p1 file
#pulls from my_dir directory if full path not given
if sys.argv[1][0] != '/':
	spect_in0 = my_dir + sys.argv[1]
else:
	spect_in0 = sys.argv[1]


#input spect p2 file
#pulls from my_dir directory if full path not given
if sys.argv[1][0] != '/':
	spect_in1 = my_dir + sys.argv[2]
else:
	spect_in1 = sys.argv[2]



#input flagp1 file
#pulls from my_dir directory if full path not given
if sys.argv[1][0] != '/':
	flag_in0 = my_dir + sys.argv[3]
else:
	flag_in0 = sys.argv[3]

#input flagp2 file
#pulls from my_dir directory if full path not given
if sys.argv[1][0] != '/':
	flag_in1 = my_dir + sys.argv[4]
else:
	flag_in1 = sys.argv[4]

print('Loading in...')
spect_in0 = np.load(spect_in0)
spect_in1 = np.load(spect_in1)
f0 = np.load(flag_in0)
f1 = np.load(flag_in1)


orig_spect_p1 = []
orig_spect_p2 = []

repl_spect_p1 = []
repl_spect_p2 = []

print('Averaging...')
for i in range(spect_in0.shape[1]):
	
	if len(spect_in0[:,i][f0[:,i]==0]) > 0:
		orig_pval_p1 = np.average(spect_in0[:,i])
		orig_pval_p2 = np.average(spect_in1[:,i])


		#excise and sum
		repl_pval_p1 = np.average(spect_in0[:,i][f0[:,i]==0][f1[:,i][f0[:,i]==0]==0])
		repl_pval_p2 = np.average(spect_in1[:,i][f0[:,i]==0][f1[:,i][f0[:,i]==0]==0])


		orig_spect_p1.append(orig_pval_p1)
		orig_spect_p2.append(orig_pval_p2)
		repl_spect_p1.append(repl_pval_p1)
		repl_spect_p2.append(repl_pval_p2)
	else:
		orig_spect_p1.append(0)
		orig_spect_p2.append(0)
		repl_spect_p1.append(0)
		repl_spect_p2.append(0)



orig_spect_p1 = np.array(orig_spect_p1)
orig_spect_p2 = np.array(orig_spect_p2)

repl_spect_p1 = np.array(repl_spect_p1)
repl_spect_p2 = np.array(repl_spect_p2)





plt.plot(orig_spect_p1,'r-',label = 'Original Pol1')
plt.plot(orig_spect_p2,'b-',label = 'Original Pol2')

plt.plot(repl_spect_p1,'r',linestyle=(0,(1,1)),label = 'Replaced Pol1')
plt.plot(repl_spect_p2,'b',linestyle=(0,(1,1)),label = 'Replaced Pol2')

plt.title('III Zwicky 35    M=4034')
plt.xlabel('Frequency Channel')

plt.legend()

plt.show()







