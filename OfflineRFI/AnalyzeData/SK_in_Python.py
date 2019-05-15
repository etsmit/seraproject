#---------------------------------------------------------
#Evan Smith
#Spectral Kurtosis Functions in python
#---------------------------------------------------------


#---------------------------------------------------------
#----Summary----
#Input data should be in numpy arrays
#Shape: (bandwidth-channels, number of integrations)
#Data is consecutive integrations 
#---------------------------------------------------------


#Imports:

#import numpy as np
#import os,sys

#import scipy as sp
#import scipy.optimize
#import scipy.special


#---------------------------------------------------------
# Functions pertaining specifically to performing SK
#---------------------------------------------------------
def SK_EST(a,n,m):
	#a=2D power spectra - shape = (bandwidth,ints)
	#n should be equal to 1                       
	#m=ints to use (from beginning of a)          
	nchans=a.shape[0]                             
	d=1#shape parameter(expect 1)                                       
	print('Nchans: ',nchans)                      
	print('M: ',m)                                                                
	print('d: ',d)                                
	#make s1 and s2 as defined by whiteboard (by 2010b Nita paper)
	#s2 definition will probably throw error if n does not integer divide m
	sum1=np.sum(a[:,:m],axis=1)                                            
	a2=a**2                                                                
	sum2=np.sum(a2[:,:m],axis=1)                                           
	#s2=sum(np.sum(a[chan,:].reshape(-1,n)**2,axis=1))#Use in case of n != 1
	sk_est = ((m*n*d+1)/(m-1))*(((m*sum2)/(sum1**2))-1)                     
	return sk_est


def upperRoot(x, moment_2, moment_3, p):
	#helps calculate upper SK threshold
	upper = np.abs( (1 - sp.special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2)))-p)
	return upper

def lowerRoot(x, moment_2, moment_3, p):
	#helps calculate lower SK threshold
	lower = np.abs(sp.special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2))-p)
	return lower

def SK_thresholds(M, N = 1, d = 1, p = 0.0013499):
	#fully calculates upper and lower thresholds
	Nd = N * d
	#Statistical moments
	moment_1 = 1
	moment_2 = ( 2*(M**2) * Nd * (1 + Nd) ) / ( (M - 1) * (6 + 5*M*Nd + (M**2)*(Nd**2)) )
	moment_3 = ( 8*(M**3)*Nd * (1 + Nd) * (-2 + Nd * (-5 + M * (4+Nd))) ) / ( ((M-1)**2) * (2+M*Nd) *(3+M*Nd)*(4+M*Nd)*(5+M*Nd))
	moment_4 = ( 12*(M**4)*Nd*(1+Nd)*(24+Nd*(48+84*Nd+M*(-32+Nd*(-245-93*Nd+M*(125+Nd*(68+M+(3+M)*Nd)))))) ) / ( ((M-1)**3)*(2+M*Nd)*(3+M*Nd)*(4+M*Nd)*(5+M*Nd)*(6+M*Nd)*(7+M*Nd) )
	#Pearson Type III Parameters
	delta = moment_1 - ( (2*(moment_2**2))/moment_3 )
	beta = 4 * ( (moment_2**3)/(moment_3**2) )
	alpha = moment_3 / (2 * moment_2)
	error_4 = np.abs( (100 * 3 * beta * (2+beta) * (alpha**4)) / (moment_4 - 1) )
	x = [1]
	upperThreshold = sp.optimize.newton(upperRoot, x[0], args = (moment_2, moment_3, p))
	lowerThreshold = sp.optimize.newton(lowerRoot, x[0], args = (moment_2, moment_3, p))
	return lowerThreshold, upperThreshold


#---------------------------------------------------------
# Functions for replacing data
#---------------------------------------------------------

#a is the input data, flags is a flag file from one of the SK_stats programs

# First to interpolate data points to flag from SK results

def 2D_overlayflags(a,f):
	#'explode' the flags file back up to the size of the original data array
	#for 2D case (not _overtime variant)
	print('Data shape: '+str(a.shape))
	print('Flags shape: '+str(f.shape))
	bigflags=np.zeros(a.shape)
	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if int(f[i,j]) == 1:
				#where in the original data does this correspond to?
				bigflags[i/2,:,i%2] = 1
				#flagging an entire coarse channel isnt very helpful huh

def 3D_overlayflags(a,f,m):
	#'explode' the flags file back up to the size of the original data array
	#for 3D case (_overtime variant)
	#m was the given SK_ints from guppi_SK_overtime.py
	print('Data shape: '+str(a.shape))
	print('Flags shape: '+str(f.shape))

	bigflags=np.zeros(a.shape)#flags file with same shape as data
	bad_datarange = m*f.shape[2]#amount of data points to flag at once in data

	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			for k in range(f.shape[2]):
				if int(f[i,j,k]) == 1:
					#where in the original data does this correspond to?
					big_i = i/2
					big_j = j*bad_datarange
					big_k = i%2
					bigflags[big_i,big_j:big_j+bad_datarange,big_k] = 1
					#currently ignores any datapoints that might have been dropped.
					#fix for this soon
	return bigflags,bad_datarange


#Next, pick a replacement method.

	
#replace all the flagged data points with zeros. Not ideal scientifically.
def zeros(a, big_f):
	out_arr = np.array(a)
	print('Replacing data with zeros')
	for i in range(big_f.shape[0]):
		for j in range(big_f.shape[1]):
			for k in range(big_f.shape[2]):
				if big_f[i,j,k] == 1:
					out_arr[i,j,k] = np.float64(0.0)
	return out_arr




#replace with previous good data (or future good)
def old_good(f,index,badrange):
	#finds previous good data to replace
	#helps keep the amount of 'tabs' in check
	if j < badrange:
		for backwards_index in range(index):
			if f[




def previous_good(a,big_f,x):
 	out_arr = np.array(a)
	print('Replacing data with zeros')
	for i in range(big_f.shape[0]):
		for j in range(big_f.shape[1]):
			for k in range(big_f.shape[2]):
				if big_f[i,j,k] == 1:
				#science
				good_data = old_good(big_f
	return out_arr





#replace with statistical noise
def statistical_noise(a,big_f):
 	out_arr = np.array(a)
	print('Replacing data with zeros')
	for i in range(big_f.shape[0]):
		for j in range(big_f.shape[1]):
			for k in range(big_f.shape[2]):
				if big_f[i,j,k] == 1:
				#science

	return out_arr



