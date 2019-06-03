#SK_stats_overtime
#input a SK result npy file and the upper and lower thresholds
#outputs a (for now) rudimentary flag mask (1 = RFI, 0 = clean)



import sys,os
import numpy as np


infile = sys.argv[1]
ut = np.float64(sys.argv[2])
lt = np.float64(sys.argv[3])
outfile = sys.argv[4]

print('Loading file...')
sk = np.load(infile)
print('SK results shape: '+str(sk.shape))


flags = np.zeros(sk.shape)
tot_points = sk.size
flagged_pts = 0

#expecting a 3D sk results array for now (overtime)
pols = sk.shape[0]
SK_timebins = sk.shape[1]
finechans = sk.shape[2]

#look at every data point
for i in range(pols):
	for j in range(SK_timebins):
		for k in range(finechans):
			
			#is the datapoint outside the threshold?
			if (sk[i,j,k] < lt) or (sk[i,j,k] > ut):
				flagged_pts += 1
				flags[i,j,k] = 1

flagged_percent = (float(flagged_pts)/tot_points)*100
print(str(flagged_pts)+' datapoints were flagged out of '+str(tot_points))
print(str(flagged_percent)+'% of data outside acceptable ranges')

np.save(outfile,flags)
print('Flags file saved to '+outfile)





