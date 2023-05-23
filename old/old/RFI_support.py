#old functions that are no longer in use
#if need to resurrect, copy/paste into ../RFI_support.py
#READ THE FUNCTION DOCUMENTATION! the function names might be the same despite doing different things


#replace with tstatistical noise
@jit(parallel=True)
def statistical_noise(a,f):
	"""
	Replace flagged data with statistical noise.
	first version
	Parameters
	-----------
	a : ndarray
		3-dimensional array of power values. Shape (Num Channels , Num Raw Spectra , Npol)
	f : ndarray
		3-dimensional array of flags. 1=RFI detected, 0 no RFI. Shape (Num Channels , Num Raw Spectra , Npol), should be same shape as a.
	
	
	Returns
	-----------
	out : np.random.normal(0,1,size=2048)ndarray
		3-dimensional array of power values with flagged data replaced. Shape (Num Channels , Num Raw Spectra , Npol)
	"""
	print('stats....')
	for pol in prange(f.shape[2]):	
		for i in prange(f.shape[0]):
			#find clean data points from same channel and polarization
			#good_data = a[i,:,pol][f[i,:,pol] == 0]
			#how many data points do we need to replace
			bad_data_size = np.count_nonzero(f[i,:,pol])
 
			#print(a[:,:,pol].shape,f[:,:,pol].shape)
			ave_real,ave_imag,std_real,std_imag = adj_chan_good_data(a[:,:,pol],f[:,:,pol],i)
			#print('generating representative awgn..')
			a[i,:,pol].real[f[i,:,pol] == 1] = np.random.normal(ave_real,std_real,bad_data_size)
			a[i,:,pol].imag[f[i,:,pol] == 1] = np.random.normal(ave_imag,std_imag,bad_data_size)
	return a



#replace with statistical noise
@jit(parallel=True)
def statistical_noise_alt(a,f,SK_M):
	"""
	Replace flagged data with statistical noise.
	Alt version that only takes the same time bin at adjacent-ish freq channels
	Parameters
	-----------
	a : ndarray
		3-dimensional array of power values. Shape (Num Channels , Num Raw Spectra , Npol)
	f : ndarray
		3-dimensional array of flags. 1=RFI detected, 0 no RFI. Shape (Num Channels , Num Raw Spectra , Npol), should be same shape as a.

	Returns
	-----------
	out : ndarray
		3-dimensional array of power values with flagged data replaced. Shape (Num Channels , Num Raw Spectra , Npol)
	"""
	print('stats....')
	print('fshape',f.shape)
	for pol in prange(f.shape[2]):
		for i in prange(f.shape[0]):
			for tb in prange(f.shape[1]//SK_M):
				if f[i,SK_M*tb,pol] == 1:
			#find clean data points from same channel and polarization
			#good_data = a[i,:,pol][f[i,:,pol] == 0]
			#how many data points do we need to replace
					bad_data_size = SK_M
 
			#print(a[:,:,pol].shape,f[:,:,pol].shape)
					ave_real,ave_imag,std_real,std_imag = adj_chan_good_data_alt(a[:,tb*SK_M:(tb+1)*SK_M,pol],f[:,tb*SK_M:(tb+1)*SK_M,pol],i,SK_M,tb)
					#print('generating representative awgn..')
					(a[i,tb*SK_M:(tb+1)*SK_M,pol].real)[f[i,tb*SK_M:(tb+1)*SK_M,pol] == 1] = np.random.normal(ave_real,std_real,SK_M)
					(a[i,tb*SK_M:(tb+1)*SK_M,pol].imag)[f[i,tb*SK_M:(tb+1)*SK_M,pol] == 1] = np.random.normal(ave_imag,std_imag,SK_M)
	#print('returning a')
	return a






