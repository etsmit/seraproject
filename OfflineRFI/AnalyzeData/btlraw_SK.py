#--------------------------------------------------
#btlraw_SK.py
#Opens one output npy file from btl_to_raw.npy and performs SK 
#Takes two inputs:
#1: input filename (BL raw format) from my_dir
#2: npy file to save to
#--------------------------------------------------
# This does SK on the entirety of the block in one go
# so no time resolution finer than the size of the block
#--------------------------------------------------

import numpy as np
import os,sys
import matplotlib.pyplot as plt

import scipy as sp
import scipy.optimize
import scipy.special


my_dir = '/home/scratch/esmith/RFI_MIT/'

#pulls from my scratch directory if full path not given
if sys.argv[1][0] != '/':
	inputFileName = my_dir + sys.argv[1]
else:
	inputFileName = sys.argv[1]

#same for  output destination
if sys.argv[2][0] != '/':
	sk_npy = my_dir + sys.argv[2]
else:
	sk_npy = sys.argv[2]



FFTLEN = int(sys.argv[3])



#--------------------------------------
# Functions
#-------------------------------------- 
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
	print('SK completed')                     
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
#--------------------------------------
#--------------------------------------





#--------------------------------------
# Fun
#--------------------------------------


print('Opening file: '+inputFileName)
data= np.load(inputFileName)
print('Data shape: '+str(data.shape))
print('#--------------------------------------')


mismatch = data.shape[1] % FFTLEN
if mismatch != 0:
	print('Warning: FFTLEN does not divide the amount of time samples')
	print(str(mismatch)+' time samples at the end will be dropped')
kept_samples = data.shape[1] - mismatch

n=1

ints = np.float64(data.shape[1]/FFTLEN)
print('With given FFTLEN, there are '+str(ints)+' spectra per polarization per coarse channel')

#calculate thresholds
print('Calculating SK thresholds...')
lt, ut = SK_thresholds(ints, N = 1, d = 1, p = 0.0013499)
print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))



print('Performing FFT...')

sk_results=[]

for i in range(32):
	for j in range(2):
		print('Coarse Channel '+str(i))
		print('Polarization '+str(j))
		data_to_FFT = data[i,:kept_samples,j]
		data_to_FFT = data_to_FFT.reshape((FFTLEN,-1))
		s = np.abs(np.fft.fft(data_to_FFT,axis=0))**2
		sk_spect = SK_EST(s,n,ints)
		sk_results.append(sk_spect)

sk_results = np.array(sk_results)
print('SK results shape: '+str(sk_results.shape))


np.save(sk_npy, sk_results)
print('SK spectra saved in '+str(sk_npy))
print('Data order: (Coarse) chan0,pol0 ; chan0,pol1 ; chan1,pol0 , etc...') 

print('Upper Threshold: '+str(ut))
print('Lower Threshold: '+str(lt))

plt.plot(sk_results[0,:],'b+')
plt.title('SK')
plt.plot(np.zeros(len(sk_results[0]))+ut, 'r-')
plt.plot(np.zeros(len(sk_results[0]))+lt, 'r-')
plt.plot(np.zeros(len(sk_results[0]))+1, 'b-')
plt.show()


plt.plot(sk_results[40,:],'b+')
plt.title('SK')
plt.plot(np.zeros(len(sk_results[0]))+ut, 'r-')
plt.plot(np.zeros(len(sk_results[0]))+lt, 'r-')
plt.plot(np.zeros(len(sk_results[0]))+1, 'b-')
plt.show()



print('Done!')












