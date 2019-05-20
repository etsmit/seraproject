#SK_stats_overtime
#input a SK result npy file and the upper and lower thresholds
#outputs a (for now) rudimentary flag mask (1 = RFI, 0 = clean)



import sys,os
import numpy as np
import commands



my_dir = '/home/scratch/esmith/RFI_MIT/'

blocks = commands.getoutput('ls '+my_dir+'SK_J1713+0747/').split('\n')



ut = np.float64(sys.argv[1])
lt = np.float64(sys.argv[2])


stats=[]


for block in blocks:
	print('Loading file '+block)
	sk = np.load(my_dir+'SK_J1713+0747/'+block)
	print('SK results shape: '+str(sk.shape))


	flags = np.zeros(sk.shape)
	tot_points = sk.size
	flagged_pts = 0

	#expecting a 3D sk results array for now (overtime)
	chans_pols = sk.shape[0]
	SK_timebins = sk.shape[1]
	finechans = sk.shape[2]

	#look at every data point
	for i in range(chans_pols):
		for j in range(SK_timebins):
			for k in range(finechans):
				
				#is the datapoint outside the threshold?
				if (sk[i,j,k] < lt) or (sk[i,j,k] > ut):
					flagged_pts += 1
					flags[i,j,k] = 1

	flagged_percent = (float(flagged_pts)/tot_points)*100
	print(str(flagged_pts)+' datapoints were flagged out of '+str(tot_points))
	print(str(flagged_percent)+'% of data outside acceptable ranges')
	stats.append(np.array([flagged_pts,tot_points,flagged_percent]))

	np.save(my_dir+'flags_J1713+0747/flags_'+block,flags)
	print('Flags file saved')

np.save('flags_tot_stats.npy',np.array(stats))




