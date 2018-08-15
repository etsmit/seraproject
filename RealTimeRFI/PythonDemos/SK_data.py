import math
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import sys

import pylab
from matplotlib import rcParams

#------------------------------------------------------------
#   INPUTS
#------------------------------------------------------------

# Arguments in order
# FileName: Name of file
# k:        Number of frequency channels (combine with time resolution to get frequency resolution)
# m:        Number of integrations (starts from 0)
# N:        Number of integrations averaged before dumping to dyn_spec
# pol:      Pol X (0) or Pol Y (1)

infile = str(sys.argv[1])
k = int(sys.argv[2])
m = int(sys.argv[3])
n = int(sys.argv[4])
pol = int(sys.argv[5])

if m*k > 1024*24570:
    sys.exit()

#------------------------------------------------------------
#   FUNCTIONS
#------------------------------------------------------------

#SK ESTIMATOR FUNCTION
# a: 2D input np array spectrograph of power.
#      Vertical as frequency channel, Horizontal as time. Shape: (numChans,numInts)
# m: number of integrations. Should be able to read from len(a)
# n: sub-sums of integrations. Will stay as 1
# d: shape parameter Will stay as 1
# Returns 1D np array of SK estimator
def SK_EST(a,n):
    m=a.shape[1]
    #m=ints
    nchans=a.shape[0]
    d=1
    print('Shape: ',a.shape)
    print('Nchans: ',nchans)
    print('M: ',m)
    print('N: ',n)
    print('d: ',d)
    #make s1 and s2 as defined by whiteboard (by 2010b)
    #s2 definition will probably throw error if n does not integer divide m
    #s1=sum(a[chan,:]) same for old s2
    s1=np.sum(a,axis=1)
    #s2=sum(np.sum(a[chan,:].reshape(-1,n)**2,axis=1))
    s2=np.sum(a**2,axis=1)
    #record sk estimator
    sk_est = ((m*n*d+1)/(m-1))*((m*s2)/(s1**2)-1)
    return sk_est



#Threshold functions
#Taken from Nick Joslyn's helpful code https://github.com/NickJoslyn/helpful-BL/blob/master/helpful_BL_programs.py
def upperRoot(x, moment_2, moment_3, p):
    upper = np.abs( (1 - scipy.special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2)))-p)
    return upper

def lowerRoot(x, moment_2, moment_3, p):
    lower = np.abs(scipy.special.gammainc( (4 * moment_2**3)/moment_3**2, (-(moment_3-2*moment_2**2)/moment_3 + x)/(moment_3/2/moment_2))-p)
    return lower

n=1
d=1

def spectralKurtosis_thresholds(M, N = n, d = d, p = 0.0013499):
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
    upperThreshold = scipy.optimize.newton(upperRoot, x[0], args = (moment_2, moment_3, p))
    lowerThreshold = scipy.optimize.newton(lowerRoot, x[0], args = (moment_2, moment_3, p))
    return lowerThreshold, upperThreshold

#------------------------------------------------------------
#   FUN
#------------------------------------------------------------


#Load chan26.npy and apply SK Estimator
#plot spectrogram

rcParams.update({'figure.autolayout' : True})
rcParams.update({'axes.formatter.useoffset' : False})

# create 512 time bins of 1024 spectra, each with 32 integrations.
# there may be a few bytes not used.
nfreq = k
nint  = n

# get the data
tsData = np.load(infile)
tsData = tsData[1100*2000:]#truncating tsData
tsLen = tsData.shape[0]

# required polarization channels
realpol = 2*pol
imagpol = realpol+1



# empty list of power spectra
spec_list = []

print(tsLen)
print(nfreq * nint)
#nspec = int(tsLen / (nfreq * nint))
nspec=m
print("processing ", str(nspec), "spectra...")


# do the work
#nspec=767 in our case
for s in range(nspec):
    print("spectrum: ", s)
    winStart = s * (nfreq * nint)
    accum = np.zeros(nfreq)
    for i in range(nint):
        start = winStart + i * nfreq
        end = start + nfreq
        in_arr = np.zeros((nfreq), dtype=np.complex_)
        in_arr.real = tsData[start:end, realpol]
        in_arr.imag = tsData[start:end, imagpol]
        out_arr = np.fft.fftshift(np.fft.fft(in_arr))
        accum += np.abs(out_arr)**2
    spec_list.append(accum/nint)


# convert back to numpy array and transpose to desired order
dyn_spec = np.transpose(np.asarray(spec_list))

plt.plot(np.average(dyn_spec,axis=1),'r-')
plt.title(infile+'_'+str(nspec)+'_'+str(k)+'_'+str(pol))
plt.savefig(infile+'_'+str(nspec)+'_'+str(k)+'_'+str(pol)+'_power.png')
plt.gcf().clear()



#------------------------------------------------------------

sk_result = SK_EST(dyn_spec,n)

lt,ut = spectralKurtosis_thresholds(np.float(nspec), N = np.float(n), d = 1, p = 0.0013499)


print 'Lower Threshold: '+str(lt)
print 'Upper Threshold: '+str(ut)

#------------------------------------------------------------

print('SK_value: '+str(np.min(sk_result)))
#print('SK_value: '+str(sk_result[193]))

plt.plot(sk_result,'b+')
plt.title(infile+'_'+str(nspec)+'_'+str(k)+'_'+str(pol))
plt.plot(np.zeros(k)+ut,'r:')
plt.plot(np.zeros(k)+lt,'r:')

plt.text(nfreq/2,(np.max(sk_result)+np.min(sk_result))/2,str(np.min(sk_result)))

plt.show()
plt.savefig(infile+'_'+str(nspec)+'_'+str(k)+'_'+str(pol)+'_sk.png')
plt.gcf().clear()

spec_mean = np.mean(dyn_spec)
spec_rms = np.std(dyn_spec)

#pylab.imshow(dyn_spec,aspect='auto',vmin=spec_mean-2*spec_rms, vmax=spec_mean+5*spec_rms)
#plt.show()


