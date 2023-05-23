#---------------------------------------------------------
"""
Evan Smith
Functions to support RFI detection methods
Detection methods found in RFI_detection


1. SK threshold calculation
	Found numerically through 'perfect' SK distribution


2. Flagged Data replacement
	-replacing data with:
		zeros
		previous good
		statistical noise

3. Supporting/misc functions




"""
#---------------------------------------------------------

import numpy as np
import os,sys

import scipy as sp
import scipy.optimize
import scipy.special

import matplotlib.pyplot as plt

from numba import jit,prange


from statsmodels.stats.weightstats import DescrStatsW





#---------------------------------------------------------
# 1 . SK threshold calculation
#---------------------------------------------------------
#TODO: citation to nita et. al. IDL code

#Thanks to Nick Joslyn for contributing Python translation

#helps calculate upper SK threshold
def upperRoot(x, moment_2, moment_3, p):
	upper = np.abs( (1 - sp.special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2)))-p)
	return upper

#helps calculate lower SK threshold
def lowerRoot(x, moment_2, moment_3, p):
	lower = np.abs(sp.special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2))-p)
	return lower

#fully calculates upper and lower thresholds
#M = SK_ints
#default p = PFA = 0.0013499 corresponds to 3sigma excision
def SK_thresholds(M, N = 1, d = 1, p = 0.0013499):
	"""
	Determine SK thresholds numerically.

	Parameters
	-----------
	m : int
		integer value of M in the SK function. Outside accumulations of spectra.
	n : int
		integer value of N in the SK function. Inside accumulations of spectra.
	d : float
		shape parameter d in the SK function. Usually 1 but can be empirically determined.
	p : float
		Prob of false alarm. 0.0013499 corresponds to 3-sigma excision.
	
	Returns
	-----------
	out : tuple
		Tuple of (lower threshold, upper threshold).
	"""

	Nd = N * d
	#Statistical moments
	moment_1 = 1
	moment_2 = float(( 2*(M**2) * Nd * (1 + Nd) )) / ( (M - 1) * (6 + 5*M*Nd + (M**2)*(Nd**2)) )
	moment_3 = float(( 8*(M**3)*Nd * (1 + Nd) * (-2 + Nd * (-5 + M * (4+Nd))) )) / ( ((M-1)**2) * (2+M*Nd) *(3+M*Nd)*(4+M*Nd)*(5+M*Nd))
	moment_4 = float(( 12*(M**4)*Nd*(1+Nd)*(24+Nd*(48+84*Nd+M*(-32+Nd*(-245-93*Nd+M*(125+Nd*(68+M+(3+M)*Nd)))))) )) / ( ((M-1)**3)*(2+M*Nd)*(3+M*Nd)*(4+M*Nd)*(5+M*Nd)*(6+M*Nd)*(7+M*Nd) )
	#Pearson Type III Parameters
	delta = moment_1 - ( (2*(moment_2**2))/moment_3 )
	beta = 4 * ( (moment_2**3)/(moment_3**2) )
	alpha = moment_3 / (2 * moment_2)
	beta_one = (moment_3**2)/(moment_2**3)
	beta_two = (moment_4)/(moment_2**2)
	error_4 = np.abs( (100 * 3 * beta * (2+beta) * (alpha**4)) / (moment_4 - 1) )
	kappa = float( beta_one*(beta_two+3)**2 ) / ( 4*(4*beta_two-3*beta_one)*(2*beta_two-3*beta_one-6) )
	print('kappa: {}'.format(kappa))
	x = [1]
	print(x, moment_2, moment_3, p)
	upperThreshold = sp.optimize.newton(upperRoot, x[0], args = (moment_2, moment_3, p))
	lowerThreshold = sp.optimize.newton(lowerRoot, x[0], args = (moment_2, moment_3, p))
	return lowerThreshold, upperThreshold


#---------------------------------------------------------
# 2 . Flagged Data Replacement
#---------------------------------------------------------



def repl_zeros(a,f):
	"""
	Replace flagged data with 0's.

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
	#these will get cast to 0 in the next step, the 1e-4 is to stop any possible issues with log10
	a[f==1]=1e-4 + 1e-4*1.j
	return a




#replace with previous good data (or future good)

def prevgood_init(a,f,x):
	#fix for missing polarization forloop
	out_arr = np.array(a)
	for pol in range(a.shape[2]):
		out_arr[:,:,pol] = previous_good(a[:,:,pol],f[:,:,pol],x)

def previous_good(a,f,x):
	"""
	Replace flagged data with copies of clean data

	Parameters
	-----------
	a : ndarray
		2-dimensional array of power values. Shape (Num Channels , Num Raw Spectra)
	f : ndarray
		2-dimensional array of flags. 1=RFI detected, 0 no RFI. Shape (Num Channels , Num Raw Spectra), should be same shape as a.
	x : int
		is just m from other functions.
	
	
	Returns
	-----------
	out : ndarray
		2-dimensional array of power values with flagged data replaced. Shape (Num Channels , Num Raw Spectra)
	"""
	out_arr = np.array(a)
	for i in range(f.shape[0]):
		#print('Coarse Chan '+str(i))
		for j in range(f.shape[1]):
			turnaround = False
			if f[i,j] == 1:
				#replace
				if (j >= 1):
					#print('Looking back at previous data')
					n=0
					while (f[i,j-n] == 1):
						if (j-n <= 0):
							#print('****No previous good data found for channel {}****'.format(i))
							turnaround=True
							break
						n += 1
					if not turnaround:
						out_arr[i,j*x:(j+1)*x] = a[i,(j-n)*x:(j-n+1)*x]

				if (j < 1) or turnaround:
					#print('Looking forward at following data')
					n=0#n=0, not 1 is redundant but necessary to keep j+n check inside loop
					while (f[i,j+n] == 1):
						n += 1
						if (j+n >= f.shape[1]):
							print('****No good data found in channel {}****'.format(i))
							out_arr = adj_chan(out_arr,f,i,x)
							break

					if (j+n >= f.shape[1]):
						break
					if (j+n < f.shape[1]):
						out_arr[i,j*x:(j+1)*x] = a[i,(j+n)*x:(j+n+1)*x]

	return out_arr





@jit(parallel=True)
def statistical_noise_alt_fir(a,f,SK_M):
	"""
	Replace flagged data with statistical noise.
	- alt version that only takes the same time bin at adjacent-ish freq channels
	- fir version that filters noise by pfb coefficients to get desired spectral response
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
	#find correct PFB coefficents
	nchan = str(f.shape[0]).zfill(4)
	#print(nchan,type(nchan))	
	hfile = '/users/esmith/RFI_MIT/PFBcoeffs/c0800x'+nchan+'_x14_7_24t_095binw_get_pfb_coeffs_h.npy'
	h = np.load(hfile)
	dec = h[::2*f.shape[0]]

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
					(a[i,tb*SK_M:(tb+1)*SK_M,pol].real)[f[i,tb*SK_M:(tb+1)*SK_M,pol] == 1] = noise_filter(ave_real,std_real,SK_M,dec)
					
					(a[i,tb*SK_M:(tb+1)*SK_M,pol].imag)[f[i,tb*SK_M:(tb+1)*SK_M,pol] == 1] = noise_filter(ave_imag,std_imag,SK_M,dec)
	#print('returning a')
	return a


#replace with statistical noise
@jit(parallel=True)
def statistical_noise_fir(a,f):
	"""
	Replace flagged data with statistical noise.
	- fir version that adds a fir in the noise
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
	#find correct PFB coefficents
	nchan = str(f.shape[0]).zfill(4)
	#print(nchan,type(nchan))	
	hfile = '/users/esmith/RFI_MIT/PFBcoeffs/c0800x'+nchan+'_x14_7_24t_095binw_get_pfb_coeffs_h.npy'
	h = np.load(hfile)
	dec = h[::2*f.shape[0]]


	for pol in prange(f.shape[2]):	
		for i in prange(f.shape[0]):
			#find clean data points from same channel and polarization
			#good_data = a[i,:,pol][f[i,:,pol] == 0]
			#how many data points do we need to replace
			bad_data_size = np.count_nonzero(f[i,:,pol])
 			#print(bad_data_size)
			#print(a[:,:,pol].shape,f[:,:,pol].shape)
			if bad_data_size > 0:
				ave_real,ave_imag,std_real,std_imag,num_iter,failed = adj_chan_good_data(a[:,:,pol],f[:,:,pol],i)
			#print('generating representative awgn..')
				a[i,:,pol].real[f[i,:,pol] == 1] = noise_filter(ave_real,std_real,bad_data_size,dec)  
				a[i,:,pol].imag[f[i,:,pol] == 1] = noise_filter(ave_imag,std_imag,bad_data_size,dec)
	return a,num_iter,failed



#replace with statistical noise
#@jit(parallel=True)
def statistical_noise_fir_abs(a,f,rms_txt,block):
	"""
	Replace flagged data with statistical noise.
	- fir version that adds a fir in the noise
	- abs version that uses the absorber data to determine correct rms
	Parameters
	-----------
	a : ndarray
		3-dimensional array of power values. Shape (Num Channels , Num Raw Spectra , Npol)
	f : ndarray
		3-dimensional array of flags. 1=RFI detected, 0 no RFI. Shape (Num Channels , Num Raw Spectra , Npol), should be same shape as a.
	rms_txt : string
		string filename that has the correct rms values from absorber tests.
	
	
	Returns
	-----------
	out : ndarray
		3-dimensional array of power values with flagged data replaced. Shape (Num Channels , Num Raw Spectra , Npol)
	"""
	print('stats....')
	print(a.shape,f.shape)
	#find correct PFB coefficents
	nchan = str(f.shape[0]).zfill(4)
	#print(nchan,type(nchan))	
	hfile = '/users/esmith/RFI_MIT/PFBcoeffs/c0800x'+nchan+'_x14_7_24t_095binw_get_pfb_coeffs_h.npy'
	h = np.load(hfile)
	dec = h[::2*f.shape[0]]

	#load absorber scan data results
	#model_chan,model_rms_x,model_rms_y = np.loadtxt(rms_txt,usecols=(0,2,3),unpack=True)
	model = np.load(rms_txt)
	model_x = model[:,0]
	model_y = model[:,1]


	#find rms and scale to match the unflagged data
	data_ma = np.copy(np.ma.masked_array(a,f))
	data_rms = np.std(data_ma,axis=1)
	fit_start = int(0.1*len(model_x))
	fit_end = int(0.65*len(model_x))
	fit_chans = np.r_[fit_start:fit_end]

	scale_x = scipy.optimize.least_squares(fitfunc,1,args=(model_x[fit_chans],data_rms[:,0][fit_chans]))["x"][0]
	scale_y = scipy.optimize.least_squares(fitfunc,1,args=(model_y[fit_chans],data_rms[:,1][fit_chans]))["x"][0]
	std_noise = np.c_[scale_x * model_x , scale_y * model_y]
	#np.save(f'/data/scratch/SKresults/absorber_rms/std_noise{block}.npy',std_noise)

	print(f.shape)
	for pol in prange(f.shape[2]):	
		for i in prange(f.shape[0]):
			#find clean data points from same channel and polarization
			#good_data = a[i,:,pol][f[i,:,pol] == 0]
			#how many data points do we need to replace
			bad_data_size = np.count_nonzero(f[i,:,pol])
 			#print(bad_data_size)
			#print(a[:,:,pol].shape,f[:,:,pol].shape)
			#because these are complex numbers....
			std_scaled = np.sqrt((std_noise[i,pol]**2)/2)
			if bad_data_size > 0:
				#ave_real,ave_imag,std_real,std_imag = adj_chan_good_data(a[:,:,pol],f[:,:,pol],i)
			#print('generating representative awgn..')
				a[i,:,pol].real[f[i,:,pol] == 1] = noise_filter(0,std_scaled,bad_data_size,dec)  
				a[i,:,pol].imag[f[i,:,pol] == 1] = noise_filter(0,std_scaled,bad_data_size,dec)
	return a


def fitfunc(a,model,data):
    return data - a*model

#alternate statistical noise generator if entire channel is flagged
#for the length of the block
#pulls unflagged data points from at most two channels on either side (less if chan c = 0,1 or ex. 254,255)


def adj_chan_good_data(a,f,c):
	"""
	Return mean/std derived from unflagged data in adjacent channels 
	Parameters
	-----------
	a : ndarray
		3-dimensional array of original power values. Shape (Num Channels , Num Raw Spectra , Npol)
	f : ndarray
		3-dimensional array of flags. 1=RFI detected, 0 no RFI. Shape (Num Channels , Num Raw Spectra , Npol), should be same shape as a.
	c : int
		Channel of interest
	
	Returns
	-----------
	ave_real : float
		average value of unflagged real data
	ave_imag : float
		average value of unflagged imaginary data
	std_real : float		
		standard deviation of unflagged real data
	std_imag : float
		standard deviation of unflagged imaginary data
	"""

	num_iter = 0
	failed = 0
	#num_iter = num_iter
	#failed = failed
	#define adjacent channels and clear ones that don't exist (neg chans, too high)
	#adj_chans = [c-3,c-2,c-1,c,c+1,c+2,c+3]
	adj_chans = [c-1,c,c+1]
	adj_chans = [i for i in adj_chans if i>=0]
	adj_chans = [i for i in adj_chans if i<a.shape[0]]

	adj_chans = np.array(adj_chans,dtype=np.uint32)

	#set up array of unflagged data and populate it with any unflagged data from adj_chans channels
	good_data=np.empty(0,dtype=np.complex64)
	good_data = np.append(good_data,a[adj_chans,:][f[adj_chans,:] == 0])

	adj=1
	#keep looking for data in adjacent channels if good_data empty
	while (good_data.size==0):
		adj += 1
		if (c-adj >= 0):
			good_data = np.append(good_data,a[c-adj,:][f[c-adj,:] == 0])
		if (c+adj < a.shape[0]):
			good_data = np.append(good_data,a[c+adj,:][f[c+adj,:] == 0])
		#if we go 8% of the spectrum away, give up and give flagged data from same channel
		failed += 1
		if adj >= int(a.shape[0]*0.08):
			good_data = a[c,:]
			break


	ave_real = np.mean(good_data.real)
	ave_imag = np.mean(good_data.imag)
	std_real = np.std(good_data.real)
	std_imag = np.std(good_data.imag)
	num_iter += 1

	return ave_real,ave_imag,std_real,std_imag,num_iter,failed




def adj_chan_good_data_alt(a,f,c,SK_M,tb):
	"""
	Return mean/std derived from unflagged data in adjacent channels
	- alt version that only takes from adjacent channel at the SAME TIME BIN
	Parameters
	-----------
	a : ndarray
		3-dimensional array of original power values. Shape (Num Channels , Num Raw Spectra , Npol)
	f : ndarray
		3-dimensional array of flags. 1=RFI detected, 0 no RFI. Shape (Num Channels , Num Raw Spectra , Npol), should be same shape as a.
	c : int
		Channel of interest
	SK_M : int
		M parameter from SK equation
	tb : int
		time bin of interest
	
	Returns
	-----------
	ave_real : float
		average value of unflagged real data
	ave_imag : float
		average value of unflagged imaginary data
	std_real : float		
		standard deviation of unflagged real data
	std_imag : float
		standard deviation of unflagged imaginary data
	"""
	#define adjacent channels and clear ones that don't exist (neg chans, too high)
	#adj_chans = [c-3,c-2,c-1,c,c+1,c+2,c+3]
	adj_chans = [c-1,c,c+1]
	adj_chans = [i for i in adj_chans if i>=0]
	adj_chans = [i for i in adj_chans if i<a.shape[0]]

	adj_chans = np.array(adj_chans,dtype=np.uint32)

	#set up array of unflagged data and populate it with any unflagged data from adj_chans channels
	good_data=np.empty(0,dtype=np.complex64)
	good_data = np.append(good_data,a[adj_chans,:][f[adj_chans,:] == 0])

	adj=1
	#keep looking for data in adjacent channels if good_data empty
	while (good_data.size==0):
		adj += 1
		if (c-adj >= 0):
			good_data = np.append(good_data,a[c-adj,:][f[c-adj,:] == 0])
		if (c+adj < a.shape[0]):
			good_data = np.append(good_data,a[c+adj,:][f[c+adj,:] == 0])
		#we gotta quit at some point, just give it flagged data from same channel
		#if we go 8% of the spectrum away
		if adj == int(a.shape[0]*0.08):
			good_data = a[c,:]
			break

	ave_real = np.mean(good_data.real)
	ave_imag = np.mean(good_data.imag)
	std_real = np.std(good_data.real)
	std_imag = np.std(good_data.imag)

	return ave_real,ave_imag,std_real,std_imag



#---------------------------------------------------------
# 3 . Supporting/misc functions
#---------------------------------------------------------

#flatten data array 'a' into format writeable to guppi file
#@jit(nopython=True)
def guppi_format(a):
	"""
	takes array of np.complex64,ravels it and outputs as 1D array of signed 8 bit integers 
	ordered x1r,x1i,y1r,y1i,x2r,x2i,y2r,....
	Parameters
	-----------
	a : ndarray
		3-dimensional array of original power values. Shape (Num Channels , Num Raw Spectra , Npol)
	Returns
	-----------
	out_arr : ndarray
		1-dimensional array of values to be written back to the copied data file
	"""
	#init output
	out_arr = np.empty(shape=2*a.size,dtype=np.int8)
	#get real values, ravel, cast to int8
	arav = a.ravel()
	a_real = np.clip(np.floor(arav.real),-128,127).astype(np.int8)
	#get imag values, ravel, cast to int8
	a_imag = np.clip(np.floor(arav.imag),-128,127).astype(np.int8)
	#interleave
	out_arr[::2] = a_real
	out_arr[1::2] = a_imag
	return out_arr



def noise_filter(ave,std,msk,dec):
	"""
	Create gaussian noise filtered by the correct PFB coefficients to mimic the VEGAS coarse channel SEFD
	Parameters
	-----------
	ave : float
		average/center value of intended noise 
	std : float
		standard deviation of noise (before FIR)
	msk : int
		M parameter of SK equation. Is also the amount of new data points to generate
	dec : decimated coefficient array to apply in FIR
	
	Returns
	-----------
	out_filtered : ndarray
		1-dimensional string of filtered gaussian noise to inject back over masked data
	"""
	#make correctly scaled noise
	out = np.random.normal(ave,std,msk)
	#do FIR
	out_filtered = np.convolve(dec,out,mode='same')
	return out_filtered


#expand small flags file (flag_chunk in guppi_SK_fromraw.py) to size of original block
#not currently used
def expand_flags(f,x):
	out_f = np.zeros((f.shape[0],f.shape[1]*x,f.shape[2]))
	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			for k in range(f.shape[2]):
				out_f[i,j*x:(j+1)*x,k] = f[i,j,k]
	return out_f

#checks replacement method inputted at top
def method_check(s):
	if s in ['zeros','previousgood','stats']:
		return True
	else:
		return False




#faster rolling window for SIR
def rollin(a,win_sz):
	shape = a.shape[:-1] + (a.shape[-1] - win_sz+1,win_sz)
	strides = a.strides + (a.strides[-1],)
	return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)
















