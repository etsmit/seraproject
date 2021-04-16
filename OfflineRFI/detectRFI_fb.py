"""
detectRFI_fb.py


Main program for detecting/excising RFI in seraproj
github.com/etsmit/seraproj

 - Opens GUPPI/VPM raw file
 - performs SK
 - gibs flags and sk files
 
 
 
 inputs:
 
 $ python detectRFI_fb.py filename SK_m is_power
 
filename : str
    input filename
    
SK_m : int
    M value for SK.
    
is_power : int
    0 = filterbank file of complex channelized voltages
    1 = squared power values


 """





import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
import scipy.optimize
import scipy.special

import sys,os


import math


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


#splits up an input data array, outputs set of SK spectra
def SK_master(s,m):
    numSKspectra = s.shape[1]//m
    print(numSKspectra)
    for i in range(numSKspectra):
        this_s = s[:,i*m:(i+1)*m]
        if i==0:
            out_sk = SK_EST(this_s,m)
            out_sk = np.expand_dims(out_sk,axis=1)
            out_s = np.expand_dims(np.mean(this_s,axis=1),axis=1)
        else:
            out_sk = np.c_[out_sk,SK_EST(this_s,m)]
            out_s = np.c_[out_s,np.mean(this_s,axis=1)]
    return out_sk,out_s
    
    
def SK_EST(a,m,n=1,d=1):
        #make s1 and s2 as defined by whiteboard (by 2010b Nita paper)
        a = a[:,:m]*n
        sum1=np.sum(a,axis=1)
        sum2=np.sum(a**2,axis=1)
        sk_est = ((m*n*d+1)/(m-1))*(((m*sum2)/(sum1**2))-1)
        return sk_est
        
        

#splits up input data array, chunks into larger MSSK bins, and performs SK
def ms_SK_master(s,m,ms0,ms1,lt,ut):
    print('---- MS SK ----')
    print(s.shape)
    numSKspectra = s.shape[1]//m
    print(numSKspectra)
    Nchan= s.shape[0]

    n=1
    d=1
    
    ms_binsize = ms0*ms1
    ms_s1 = np.zeros((s.shape[0]-(ms0-1),numSKspectra-(ms1-1)))
    ms_s2 = np.zeros((s.shape[0]-(ms0-1),numSKspectra-(ms1-1)))
    
    #fill single scale s1,s2
    for i in range(numSKspectra):
        this_s = s[:,i*m:(i+1)*m]
        if i==0:
            s1 = np.expand_dims(np.sum(this_s,axis=1),axis=1)
            s2 = np.expand_dims(np.sum(this_s**2,axis=1),axis=1)
            
        else:
            s1 = np.c_[s1,np.sum(this_s,axis=1)]
            s2 = np.c_[s2,np.sum(this_s**2,axis=1)]
                  
    print(s1.shape)
    #fill multiscale s1, s2
    for ichan in range(ms0):
        for itime in range(ms1):
            
            ms_s1 += (1./ms_binsize) * (s1[ichan:ichan+(Nchan-(ms0-1)),itime:itime+(numSKspectra-(ms1-1))])
            ms_s2 += (1./ms_binsize) * (s2[ichan:ichan+(Nchan-(ms0-1)),itime:itime+(numSKspectra-(ms1-1))])
            
            
            #ms_s1 += (s1[ichan:ichan+(Nchan-(ms0-1)),itime:itime+(numSKspectra-(ms1-1))])
            #ms_s2 += (s2[ichan:ichan+(Nchan-(ms0-1)),itime:itime+(numSKspectra-(ms1-1))])
    print(ms_s1.shape)
            
    #plt.imshow(np.log10(ms_s1),interpolation='nearest',aspect='auto',cmap='hot',vmin=2.5,vmax=3)
    #plt.colorbar()
    #plt.show()


    #Multiscale SK
    for k in range(numSKspectra-(ms1-1)):


        #sk_spect = ms_SK_EST(ms_s1[:,k],ms_s2[:,k],numSKspectra-(ms1-1),n,d)
        sk_spect = ms_SK_EST(ms_s1[:,k],ms_s2[:,k],m,n,d)
        #sk_spect[:,1] = ms_SK_EST(ms_s1[:,k],ms_s2[:,k],numSKspectra-(ms1-1),n,d)

        
        ms_flag_spect = np.zeros((Nchan-(ms0-1)),dtype=np.int8)
        
        ms_flag_spect[sk_spect>ut] = 1
        ms_flag_spect[sk_spect<lt] = 1

        #append to results
        if (k==0):
            ms_sk_block=np.expand_dims(sk_spect,axis=1)
            ms_flags_block = np.expand_dims(ms_flag_spect,axis=1)

        else:
            ms_sk_block=np.c_[ms_sk_block,np.expand_dims(sk_spect,axis=1)]
            ms_flags_block = np.c_[ms_flags_block,np.expand_dims(ms_flag_spect,axis=1)]
    print('----')
    
              
    return ms_sk_block,ms_flags_block,ms_s1
    
    
def ms_SK_EST(s1,s2,m,n=1,d=1):
    sk_est = ((m*n*d+1)/(m-1))*(((m*s2)/(s1**2))-1)

    return sk_est
    
    
    
#===================================================================================================

#parse inputs
input_filename = sys.argv[1]

SK_m = int(sys.argv[2])

is_power = bool(int(sys.argv[3]))

if is_power:
    data = np.load(input_filename)
else:
    fb = np.load(input_filename)
    data = np.abs(fb)**2




#s32_p1 = s32[:,:,0]
#s = s32_p1

fb = None

#s = np.abs(fb)**2

#hann_s = np.abs(hann_s)**2
#s = np.abs(s.T)**2
print(data.shape)


#find data shape
Nchan = data.shape[0]
Nspectra = data.shape[1]


#multiscale SK
ms0 = 1 # amount of channels in MS-SK
ms1 = 1 # amount of time pixels in MS-SK

#find flagging thresholds
lt,ut=SK_thresholds(SK_m)
print(lt)
print(ut)


#how many SK bins do we have?
num_SK_spectra = Nspectra//SK_m




#are there two polarizations?
#if len(s.shape) == 3:
#    Npol=2
#elif = len(s.shape) == 2:
#    Npol=1


#initialize output arrays
s_all = np.zeros((Nchan,num_SK_spectra,2))
sk_all = np.zeros((Nchan,num_SK_spectra,2))
f_all = np.zeros((Nchan,num_SK_spectra,2))
mssk_all = np.zeros((Nchan-(ms0-1),num_SK_spectra-(ms1-1),2))



for i in range(2):#each polarization

    #find single-scale SK
    sk,s_ave = SK_master(data[:,:,i],SK_m)

    #find multiscale SK
    ms_sk,ms_f,ms_s1 = ms_SK_master(data[:,:,i],SK_m,ms0,ms1,lt,ut)


    #flag from single-scale SK results
    f = np.zeros(sk.shape)
    f[sk>ut]=1
    f[sk<lt]=1

    f_sing = np.array(f)
    

    #add in multiscale SK results to flag array
    for ichan in range(ms0):
        for itime in range(ms1):
            f[ichan:ichan+(Nchan-(ms0-1)),itime:itime+(num_SK_spectra-(ms1-1))][ms_f==1] = 1
        
    
    #apply flags to power spectra
    #s_flag_both = np.array(s_ave)
    #s_flag_both[f==1]=1e-3

    #s_flag_sing = np.array(s_ave)
    #s_flag_sing[f_sing==1]=1e-3



    #add to final results
    sk_all[:,:,i] = sk
    mssk_all[:,:,i] = ms_sk
    f_all[:,:,i] = f
    s_all[:,:,i] = s_ave

    m = np.arange(Nchan)*800/Nchan
    ext = [0,100,50,0]

#fname = 'ask_{}ksps_chan{}_m{}_dc{}_{}ms_bias{}_spect.png'.format(sym_rate,cc_sig,SK_m,dc,dcper,ask_bias)


#plt.imshow(np.log10(s_ave),interpolation='nearest',aspect='auto',cmap='hot',extent=ext,vmin=2.5,vmax=3)
plt.imshow(np.log10(s_ave),interpolation='nearest',aspect='auto',cmap='hot',vmin=2.1,vmax=3.2)
#plt.imshow(np.log10(s_ave),interpolation='nearest',aspect='auto',cmap='hot')
#plt.ylim((140,100))
plt.ylabel('Channel')
plt.xlabel('Time')
plt.colorbar()
#plt.title(fname)
plt.tight_layout()
#plt.savefig(save_dir+fname,format='png')
plt.show()



flagged_pct_p1 = np.around((100.*np.count_nonzero(f_all[:,:,0]))/f_all[:,:,0].size,2)
flagged_pct_p2 = np.around((100.*np.count_nonzero(f_all[:,:,1]))/f_all[:,:,1].size,2)


print('% Flagged\n p1: {} || p2: {}'.format(flagged_pct_p1,flagged_pct_p2))

#apply flags from both pols to both pols
f_all[:,:,0][f_all[:,:,1]==1] = 1
f_all[:,:,1][f_all[:,:,0]==1] = 1

#this gets calculated after union of flags applied
flagged_pct_both = np.around((100.*np.count_nonzero(f_all[:,:,1]))/f_all[:,:,1].size,2)
print('Union of pols: {}'.format(flagged_pct_both))

base_output_fname = input_filename[:-4]

#save output arrays
s_all_fname = base_output_fname+'_spect.npy'
np.save(s_all_fname, s_all)

f_all_fname = base_output_fname+'_flags.npy'
np.save(f_all_fname, f_all)

sk_all_fname = base_output_fname+'_sk.npy'
np.save(sk_all_fname, sk_all)

mssk_all_fname = base_output_fname+'_mssk.npy'
np.save(mssk_all_fname, mssk_all)


