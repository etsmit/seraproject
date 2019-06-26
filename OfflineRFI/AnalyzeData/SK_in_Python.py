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

import numpy as np
import os,sys

import scipy as sp
import scipy.optimize
import scipy.special


#---------------------------------------------------------
# Functions pertaining specifically to performing SK
#---------------------------------------------------------
def SK_EST(a,n,m):
	#a=2D power spectra - shape = (bandwidth,ints)
	#n should be equal to 1                       
	#m=ints to use (from beginning of a)          
	nchans=a.shape[0]                             
	d=1#shape parameter(expect 1)                                       
	#print('Nchans: ',nchans)                      
	#print('M: ',m)                                                                
	#print('d: ',d)                                
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
# Function for fixing offset at beginning of blocks
#---------------------------------------------------------

#takes all the rows down to where the shift ends, and shifts them to the left to match the rest of the file
#last chan doesn't get shifted - so the last two coarse chans are duplicates
def offset_Fix(data_block,blocknum,offset_amt):
	shift_end = offset_amt*(1+blocknum)
	nchan = data_block.shape[0]

	for i in range(2):
		for chan in range(nchan-1):
			data_block[chan,:shift_end,i] = data_block[chan+1,:shift_end,i]
	return data_block



#---------------------------------------------------------
# Functions for replacing data
#---------------------------------------------------------
#meant to be used inline inside guppi_SK_fromraw.py with numpy arrays

#a is the input array, f is flags array, x is SK_ints


def expand_flags(f,x):
	#expand small flags file (flag_chunk in guppi_SK_fromraw.py) to size of original block
	#not currently used
	out_f = np.zeros((f.shape[0],f.shape[1]*x,f.shape[2]))
	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			for k in range(f.shape[2]):
				out_f[i,j*x:(j+1)*x,k] = f[i,j,k]
	return out_f

def method_check(s):
	#checks replacement method
	if s in ['zeros','previousgood','stats']:
		return True
	else:
		return False

def guppi_format(a):
	#flatten data array 'a' into format writeable to guppi file
	print('Creating new array...')
	#out_arr = np.zeros((a.shape[0],a.shape[1],a.shape[2]*2))
	print('Writing to it...')
	out_arr = a.view(np.float32)
	#out_arr[:,:,::2] = a.real
	#out_arr[:,:,1::2] = a.imag
	print('Flattening...')
	out_arr = out_arr.ravel()
	print('Re-formatting...')
	out_arr = out_arr.astype(np.int8)
	return out_arr

#Next, pick a replacement method.

	
#replace all the flagged data points with zeros. Not ideal scientifically.
def repl_zeros(a,f,x,p):
	out_arr = np.array(a)
	print('Replacing pol{} flagged data with zeros'.format(p))
	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f[i,j] == 1:
				out_arr[i,j*x:(j+1)*x] = np.float64(0.0)
	
	return out_arr




#replace with previous good data (or future good)
def previous_good(a,f,x,p):
	# f is the smaller flags array - flag_chunk in guppi_SK_fromraw.py
	#x is the amount of data needed (should be SK_ints) 

	out_arr = np.array(a)
	print('Replacing pol{} flagged data with previous good data'.format(p))
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




#replace with statistical noise
def gen_good_data(a,f,x,i):
	good_data = []
	#print('Coarse Chan '+str(i))
	for j in range(f.shape[1]):
		#create good data to pull noise stats from
		if f[i,j] == 0:
			good_data.append(a[i,j*x:(j+1)*x])

	good_data = np.array(good_data).flatten()
	return good_data



def statistical_noise(a,f,x,p):
	#x is the amount of data needed (bad_datarange from 3D_overlayflags)
 	#out_arr = np.array(a)
	print('Replacing Pol{} data with statistical noise'.format(p))
	#print(a.shape)
	#print(f.shape)
	#print(x)
	for i in range(f.shape[0]):
		#print('coarsechan {}'.format(i))
		good_data = gen_good_data(a,f,x,i)
 
		repl_chunk = np.zeros(x,dtype=np.complex64)

		if len(good_data) == 0:
			print('****No good data in channel {}****'.format(i))
			a = adj_chan(a,f,i,x)
		elif len(good_data) < (2*x+1):
			print('****Low number of good data in channel {} : {} data points****'.format(i,len(good_data)))
		else:
			ave_real = np.mean(good_data.real)
			ave_imag = np.mean(good_data.imag)
			std_real = np.std(good_data.real)
			std_imag = np.std(good_data.imag)
			for y in range(f.shape[1]):
				if f[i,y] == 1:
					#science
					#print(y)
					repl_chunk.real = np.random.normal(ave_real,std_real,x)
					repl_chunk.imag = np.random.normal(ave_imag,std_imag,x)
					a[i,y*x:(y+1)*x] = repl_chunk

	return a

def adj_chan(a,f,c,x):
	#replace a bad channel 'c' with stat. noise derived from adjacent channels
	out_arr = np.array(a)
	good_data = []

	adj_chans = [c-2,c-1,c+1,c+2]
	adj_chans = [i for i in adj_chans if i>=0]
	adj_chans = [i for i in adj_chans if i<a.shape[1]]
 	#print('Pulling data from channels: {}'.format(adj_chans))
	
	for i in adj_chans:
		good_data.extend(list(gen_good_data(out_arr,f,x,i)))
	good_data = np.array(good_data).flatten()

	ave_real = np.mean(good_data.real)
	ave_imag = np.mean(good_data.imag)
	std_real = np.std(good_data.real)
	std_imag = np.std(good_data.imag)

	out_arr[c,:].real = np.random.normal(ave_real,std_real,out_arr.shape[1])
	out_arr[c,:].imag = np.random.normal(ave_imag,std_imag,out_arr.shape[1])
	
	return out_arr





