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
	error_4 = np.abs( (100 * 3 * beta * (2+beta) * (alpha**4)) / (moment_4 - 1) )
	x = [1]
	upperThreshold = sp.optimize.newton(upperRoot, x[0], args = (moment_2, moment_3, p))
	lowerThreshold = sp.optimize.newton(lowerRoot, x[0], args = (moment_2, moment_3, p))
	return lowerThreshold, upperThreshold


#---------------------------------------------------------
# Functions for fixing offset at beginning of blocks
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


#Next, pick a replacement method.

	
#replace all the flagged data points with zeros. Not ideal scientifically.
def repl_zeros(a,f,x):

	print('---------------------------------------------')
	out_arr = np.array(a)
	print('Replacing flagged data with zeros')
	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			for k in range(f.shape[2]):
				if f[i,j,k] == 1:
					out_arr[i,j*x:(j+1)*x,k] = np.float64(0.0)
	print('---------------------------------------------')
	return out_arr




#replace with previous good data (or future good)
def previous_good(a,f,x):
	# f is the smaller flags array - flag_chunk in guppi_SK_fromraw.py
	#x is the amount of data needed (should be SK_ints) 
	print('---------------------------------------------')
	out_arr = np.array(a)
	print('Replacing flagged data with previous good data')
	for i in range(f.shape[0]):
		print('Coarse Chan '+str(i))
		for j in range(f.shape[1]):
			for k in range(f.shape[2]):
				print('Pol '+str(k))
				if f[i,j,k] == 1:
				#replace
					if (j >= 1):
						print('Looking back at previous data')
						n=1
						while (f[i,j-n,k] == 1):
							if (j-n < 0):
								print('No previous good data found')
								break
							n += 1
						print('Replacing data from '+str(n)+'dataranges back')
						out_arr[i,j*x:(j+1)*x,k] = a[i,j-(n+1)*x:j-n*x,k]

					if (j < 1):
						print('Looking forward at following data')
						good_data = old_good(big_f[i,j+x,k])
						n=1
						while (f[i,j+n,k] == 1):
							if (j+n >= f.shape[1]):
								print('No good data found')
								break
							n += 1
						out_arr[i,j*x:(j+1)*x,k] = a[i,j+n*x:j+(n+1)*x,k]
						if (j+n >= f.shape[1]):
							print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
							print('Coarse chan: '+str(i)+' Pol: '+str(k)+'|| Entire channel is flagged')

							print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
	print('---------------------------------------------')
	return out_arr





#replace with statistical noise
def statistical_noise(a,f,x):
	#x is the amount of data needed (bad_datarange from 3D_overlayflags)
	print('---------------------------------------------')
 	out_arr = np.array(a)
	print('Replacing data with zeros')
	for i in range(f.shape[0]):
		print('Coarse Chan '+str(i))
		for j in range(0,f.shape[1],x):
			for k in range(f.shape[2]):
				print('Pol '+str(k))

				#create good data to pull noise stats from
				good_data = []
				for y in range(f.shape[1]):
					if big_f[i,y,k] == 0:
						good_data.append(out_arr[i,y*x:(y+1)*x,k])
				good_data = np.array(good_data)
				print(str(len(good_data))+' good data points')
				ave = np.average(good_data)
				std = np.std(good_data)

				if big_f[i,j,k] == 1:
					#science
					print(j)
					out_arr[i,j*x:(j+1)*x,k] = np.random.normal(ave,std,x)
	print('---------------------------------------------')
	return out_arr



