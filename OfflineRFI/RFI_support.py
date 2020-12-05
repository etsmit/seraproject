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

#from numba import jit








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
	#these will get cast to 0 in the next step, the 1e-4 is to stop issues with log10
	a[f==1]=1e-4 + 1e-4*1.j
	return a




#replace with previous good data (or future good)

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




#supports statistical_noise function
#i = coarse channel of interest

def gen_good_data(a,f,x,i):
	
	good_data
	good_data = []
	#print('Coarse Chan '+str(i))
	for j in range(f.shape[1]):
		#create good data to pull noise stats from
		if f[i,j] == 0:
			good_data.append(a[i,j*x:(j+1)*x])

	good_data = np.array(good_data).flatten()
	return good_data


#replace with statistical noise

def statistical_noise(a,f,x):
	"""
	Replace flagged data with statistical noise.

	Parameters
	-----------
	a : ndarray
		3-dimensional array of power values. Shape (Num Channels , Num Raw Spectra , Npol)
	f : ndarray
		3-dimensional array of flags. 1=RFI detected, 0 no RFI. Shape (Num Channels , Num Raw Spectra , Npol), should be same shape as a.
	x : int
		is just m from other functions.
	
	
	Returns
	-----------
	out : ndarray
		3-dimensional array of power values with flagged data replaced. Shape (Num Channels , Num Raw Spectra , Npol)
	"""
	for pol in range(f.shape[2]):	
		for i in range(f.shape[0]):
			#find clean data points from same channel and polarization
			good_data = a[i,:,pol][f[i,:,pol] == 0]
			#how many data points do we need to replace
			bad_data_size = a[i,:,pol][f[i,:,pol] == 1].size
 
			repl_chunk = np.zeros(x,dtype=np.complex64)

			if len(good_data) == 0:
				print('****No good data in channel {}****'.format(i))
			#-next line replaces data-
				#taking this out for now, the current adj_chans is not enough
				#leaving this data as is
				a = adj_chan(a,f,i,x)
			elif len(good_data) < (2*x+1):
				print('****Low number of good data in channel {} : {} data points****'.format(i,len(good_data)))
			else:
				ave_real = np.mean(good_data.real)
				ave_imag = np.mean(good_data.imag)
				std_real = np.std(good_data.real)
				std_imag = np.std(good_data.imag)

				a[i,:,pol][f[i,:,pol] == 1].real = np.random.normal(ave_real,std_real,bad_data_size).astype(np.int8)
				a[i,:,pol][f[i,:,pol] == 1].imag = np.random.normal(ave_imag,std_imag,bad_data_size).astype(np.int8)
						#a[i,y*x:(y+1)*x] = repl_chunk

	return a


#alternate statistical noise generator if entire channel is flagged
#for the length of the block
#pulls unflagged data points from at most two channels on either side (less if chan c = 0,1 or ex. 254,255)


def adj_chan(a,f,c,x):
	#replace a bad channel 'c' with stat. noise derived from adjacent channels

	for p in pol:
		out_arr = np.array(a)
		good_data = []
		good_data = a[i,:,pol][f[i,:,pol] == 0]

		#define adjacent channels and clear ones that don't exist
		adj_chans = [c-2,c-1,c+1,c+2]
		adj_chans = [i for i in adj_chans if i>=0]
		adj_chans = [i for i in adj_chans if i<a.shape[1]]
 		#print('Pulling data from channels: {}'.format(adj_chans))
	
		for i in adj_chans:
			good_data.extend(good_data = a[i,:,pol][f[i,:,pol] == 0])
		good_data = np.array(good_data).flatten()

		ave_real = np.mean(good_data.real)
		ave_imag = np.mean(good_data.imag)
		std_real = np.std(good_data.real)
		std_imag = np.std(good_data.imag)
		
		a[c,:,pol][f[i,:,pol] == 1].real = np.random.normal(ave_real,std_real,bad_data_size).astype(np.int8)
		a[c,:,pol][f[i,:,pol] == 1].imag = np.random.normal(ave_imag,std_imag,bad_data_size).astype(np.int8)
		

		#out_arr[c,:].real = np.random.normal(ave_real,std_real,out_arr.shape[1])
		#out_arr[c,:].imag = np.random.normal(ave_imag,std_imag,out_arr.shape[1])
	
	return a



#---------------------------------------------------------
# 3 . Supporting/misc functions
#---------------------------------------------------------
#meant to be used inline inside guppi_SK_fromraw.py with numpy arrays

#INPUTS:
#a is the input array
#f is flags array
#x is SK_ints

#First, some supporting functions:

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

#flatten data array 'a' into format writeable to guppi file

def guppi_format(a):
	#takes array of np.complex64,ravels it and outputs as 1D array of signed
	#8 bit integers ordered real,imag,real,imag,.....
	#init output
	out_arr = np.empty(shape=2*a.size,dtype=np.int8)
	#get real values, ravel, cast to int8
	a_real = a.ravel().real.astype(np.int8)
	#get imag values, ravel, cast to int8
	a_imag = a.ravel().imag.astype(np.int8)
	#interleave
	out_arr[::2] = a_real
	out_arr[1::2] = a_imag
	return out_arr


#faster rolling window for SIR
def rollin(a,win_sz):
	shape = a.shape[:-1] + (a.shape[-1] - win_sz+1,win_sz)
	strides = a.strides + (a.strides[-1],)
	return np.lib.stride_tricks.as_strided(a,shape=shape,strides=strides)
















