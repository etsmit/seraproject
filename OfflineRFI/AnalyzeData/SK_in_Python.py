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








