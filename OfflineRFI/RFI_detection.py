#---------------------------------------------------------
"""
Evan Smith
Functions to detect RFI
Supporting / misc functions in RFI_support


1. Spectral Kurtosis



2. Spectral Entropy/Relative Spectral Entropy




"""
#---------------------------------------------------------


import numpy as np
import os,sys

from numba import jit



#---------------------------------------------------------
# 1 . Functions for performing SK
#---------------------------------------------------------

#Compute SK on a 2D array of power values

def SK_EST(a,m,n=1,d=1):
	"""
	Compute SK on a 2D array of power values.

	Parameters
	-----------
	a : ndarray
		2-dimensional array of power values. Shape (Num Channels , Num Raw Spectra)
	n : int
		integer value of N in the SK function. Inside accumulations of spectra.
	m : int
		integer value of M in the SK function. Outside accumulations of spectra.
	d : float
		shape parameter d in the SK function. Usually 1 but can be empirically determined.
	
	Returns
	-----------
	out : ndarray
		Spectrum of SK values.
	"""

	#make s1 and s2 as defined by whiteboard (by 2010b Nita paper)
	a = a[:,:m]*n
	sum1=np.sum(a,axis=1)
	sum2=np.sum(a**2,axis=1)
	sk_est = ((m*n*d+1)/(m-1))*(((m*sum2)/(sum1**2))-1)                     
	return sk_est





#multiscale variant
#only takes n=1 for now
#takes sum1 and sum2 as arguments rather than computing inside
def ms_SK_EST(s1,s2,m):
	"""
	Multi-scale Variant of SK_EST.

	Parameters
	-----------
	s1 : ndarray
		2-dimensional array of power values. Shape (Num Channels , Num Raw Spectra)

	s2 : ndarray
		2-dimensional array of squared power values. Shape (Num Channels , Num Raw Spectra)

	m : int
		integer value of M in the SK function. Outside accumulations of spectra.

	
	Returns
	-----------
	out : ndarray
		Spectrum of SK values.
	"""
	n=1
	d=1
	sk_est = ((m*n*d+1)/(m-1))*(((m*s2)/(s1))-1)
	return sk_est





#---------------------------------------------------------
# 2 . Functions pertaining to Spectral Entropy
#---------------------------------------------------------



def entropy(a,nbit):
	"""
	Spectral Entropy RFI detection (dev. by Natalia Schmidt, Thirimachos Bourlai)

	Parameters
	-----------
	a : ndarray
		2-dimensional array of power values. Shape (Num Channels , Num Raw Spectra)
	nbit : int
		number of bits that power values range across. (ex. 0-255 : nbit=8)

	
	Returns
	-----------
	out : ndarray
		Spectrum of spectral entropy values.
	"""
	nchan = a.shape[0]
	H = np.empty(nchan,dtype = np.float32)
	for i in range(nchan):
		num,_ = plt.hist(a[i,:],bins=range(0,2**nbit,1))
		num /= np.sum(num)
		H[i] = np.sum(-num*np.log2(num))
	return H



























