#---------------------------------------------------------
"""
Evan Smith
Functions to detect RFI
Supporting / misc functions in RFI_support


1. Spectral Kurtosis

2. Spectral Entropy/Relative Spectral Entropy

3 . Scale-invariant Rank operator


"""
#---------------------------------------------------------


import numpy as np
import os,sys

import time

import matplotlib.pyplot as plt

from RFI_support import *

from numba import jit,prange



#---------------------------------------------------------
# 1 . Functions for performing SK
#---------------------------------------------------------

#Compute SK on a 2D array of power values

@jit(parallel=True)
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


@jit(parallel=True)
def SK_EST_alt(s1,s2,m,n=1,d=1):
	"""
	Compute SK on a 2D array of power values, using s1 and s2 given instead of data

	Parameters
	-----------

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
	sk_est = ((m*n*d+1)/(m-1))*(((m*s2)/(s1**2))-1)                     
	return sk_est






#multiscale variant
#only takes n=1 for now
#takes sum1 and sum2 as arguments rather than computing inside
@jit(parallel=True)
def ms_SK_EST(s1,s2,m,n=1,d=1):
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

	ms0 : int
		axis 0 multiscale
	
	ms1 : int
		axis 1 multiscale
	
	Returns
	-----------
	out : ndarray
		Spectrum of SK values.
	"""
	
                 #((m*n*d+1)/(m-1))*(((m*sum2)/(sum1**2))-1)
	sk_est = ((m*n*d+1)/(m-1))*(((m*s2)/(s1**2))-1)
	#print(sk_est)
	return sk_est


#secondary SK variant
#takes mean/std of unflagged data points in 7 nearest chans (3 on each side)
#flags anything mean+2*std away in log power AND sk
def adj_chan_skflags(a,f,sk,a_sig,sk_sig):
	
	#powers working in log space
	a[a==0]=1e-3
	loga = np.log10(a)

	for pol in prange(a.shape[2]):
		for c in range(a.shape[0]):

			#define adjacent channels and clear ones that don't exist
			adj_chans = [c-4,c-3,c-2,c-1,c+1,c+2,c+3,c+4]
			adj_chans = [i for i in adj_chans if i>=0]
			adj_chans = [i for i in adj_chans if i<a.shape[1]]

			#find 'clean' points not flagged by SK
			clean_a = a[adj_chans,:,:][f[adj_chans,:,:]==0]
			clean_sk = sk[adj_chans,:,:][f[adj_chans,:,:]==0]

			a_thresh = np.mean(clean_a)+a_sig*np.std(clean_a)
			sk_thresh = np.mean(clean_sk)+sk_sig*np.std(clean_sk)

			#add flags based on mean+sig*sigma
			f[c,:,:][a[c,:,:]>a_thresh]=1
			f[c,:,:][sk[c,:,:]>sk_thresh]=1

	return f


#---------------------------------------------------------
# 2 . Functions pertaining to Spectral Entropy
#---------------------------------------------------------



def entropy(a,nbit=16):
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
		num,bins,patches = plt.hist(a[i,:],bins=range(0,2**nbit,1))
		np.save('num.npy',num)
		#print(type(x))
		#print(x)
		num /= np.sum(num)
		H[i] = np.sum(-num*np.log2(num))
	plt.cla()
	return H



#---------------------------------------------------------
# 3 . Scale-invariant Rank operator - extends flags
#---------------------------------------------------------




#combine: 'union', 'vert', 'hori'
def sir(a,row_eta,col_eta,combine):
	"""
	SIR operator - takes a input flagging mask and extend them in time/freq.
	eta values determine how liberal extra flagging is. 1 = whole mask turns into flags, 0=no extra flags added.

	Parameters
	-----------
	a : ndarray
		2-dimensional array of mask bools assuming 1=flag,0=clean. Shape (Num Channels , Num Raw Spectra)
	row_eta : float
		float in range (0-1) to determine how many extra rows are flagged
	col_eta : float
		float in range (0-1) to determine how many extra columns are flagged
	combine : str
		must be chosen from ['union', 'vert', 'hori']. How column and row flags are combined.

	Returns
	-----------
	out : ndarray
		2-dimensional array of extended mask bools
	"""
	t_start = time.time()
	print(t_start)

	max_row_win = 0.01
	max_col_win = 1
    
	sir_mask = np.array(a)
	row_mask = np.zeros(a.shape)
	col_mask = np.zeros(a.shape)

	if combine == 'union':
		print('doing rows...')
		for row in range(a.shape[0]):
			row_mask[row,:] = sir_1d(a[row,:],row_eta,max_row_win)
		print('doing cols...')
		for col in range(a.shape[1]):
			col_mask[:,col] = sir_1d(a[:,col],col_eta,max_col_win)
		sir_mask[row_mask==2]=2
		sir_mask[col_mask==2]=2
		#reset original flags back to 1's
		#sir_mask[a==1]=1

	if combine == 'vert':
		for col in range(a.shape[1]):
			col_mask[:,col] = sir_1d(a[:,col],col_eta,max_col_win)
		col_mask[col_mask==2]=1
		for row in range(a.shape[0]):
			row_mask[row,:] = sir_1d(col_mask[row,:],row_eta,max_row_win)
		sir_mask[row_mask==2]=3
		sir_mask[col_mask==2]=2
		#reset original flags back to 1's
		#sir_mask[a==1]=1
        
	if combine == 'hori':
		for row in range(a.shape[0]):
			row_mask[row,:] = sir_1d(a[row,:],row_eta,max_row_win)
		row_mask[row_mask==2]=1
		for col in range(a.shape[1]):
			col_mask[:,col] = sir_1d(row_mask[:,col],col_eta,max_col_win)
		sir_mask[col_mask==2]=3
		sir_mask[row_mask==2]=2
		#reset original flags back to 1's
		#sir_mask[a==1]=1

	#set all flags to 1
	#(this isn't done in the meat of the function in the case you want to compare original to extended)
	sir_mask[sir_mask != 0] = 1

	
	t_end = time.time()
	print('sir took {} seconds'.format(t_end-t_start))
	return sir_mask



def sir_1d(a,eta,max_win_sz):
	"""
	SIR in 1D, for use in sir() above

	Parameters
	-----------
	a : ndarray
		1-dimensional array of mask bools assuming 1=flag,0=clean. Shape (Num Channels , Num Raw Spectra)
	eta : float
		float in range (0-1) to determine how many extra rows are flagged
	max_win_sz : int
		maximum window size. Speeds up the code if you don't want to flag huge sections of data

	Returns
	-----------
	out : ndarray
		1-dimensional array of extended mask bools
	"""
	out = np.array(a)
	a_sz = len(a)
	for win_sz in range(2,int(a_sz*max_win_sz)):
		i=0
		r = rollin(a,win_sz)
		flg = np.count_nonzero(r,axis=1)
		[flg >= (1-eta)*win_sz]
		for window in r:
			flg = np.count_nonzero(window)
			if flg >= (1-eta)*win_sz:
				out[i:i+win_sz] = 2
			i += 1
	return out






















