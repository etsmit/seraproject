#shows SK spectra one after another

import sys
import matplotlib.pyplot as plt
import numpy as np



infile = sys.argv[1]
lt = np.float64(sys.argv[2])
ut = np.float64(sys.argv[3])

sk_results = np.load(infile)

for i in range(sk_results.shape[1]):
	plt.plot(sk_results[i,:],'b+')
	plt.title('Coarse Chan '+str(i/2)+' ; Pol '+str(i%2))
	plt.plot(np.zeros(len(sk_results[0]))+ut, 'r-')
	plt.plot(np.zeros(len(sk_results[0]))+lt, 'r-')
	plt.plot(np.zeros(len(sk_results[0]))+1, 'b-')
	plt.show()