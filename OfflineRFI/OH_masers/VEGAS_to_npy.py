#--------------------------------------
#VEGAS_to_npy.py
#Takes VEGAS raw files (data from ABELL370) and saves them as npy for SK
#Inputs:
#1: datafile to load
#2: npy file to save to (with .npy attached)
#--------------------------------------

import numpy as np
import os,sys
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
%matplotlib inline



infile = sys.argv[1]
outfile = sys.argv[2]



print('Loading '+infile)
hdulist = pyfits.open(infile)
datatable = hdulist['SINGLE DISH'].data
data = datatable.field('DATA')
print('Data Loaded. Reshaping...')

#in the order they appear in the data
if1_pl1_cd0=[]
if1_pl1_cd1=[]
if1_pl0_cd0=[]
if1_pl0_cd1=[]
if0_pl1_cd0=[]
if0_pl1_cd1=[]
if0_pl0_cd0=[]
if0_pl0_cd1=[]
if2_pl1_cd0=[]
if2_pl1_cd1=[]
if2_pl0_cd0=[]
if2_pl0_cd1=[]
if3_pl1_cd0=[]
if3_pl1_cd1=[]
if3_pl0_cd0=[]
if3_pl0_cd1=[]

for i in range(0,32800,16):
    if1_pl1_cd0.append(data[i+0,:])
    if1_pl1_cd1.append(data[i+1,:])
    if1_pl0_cd0.append(data[i+2,:])
    if1_pl0_cd1.append(data[i+3,:])
    if0_pl1_cd0.append(data[i+4,:])
    if0_pl1_cd1.append(data[i+5,:])
    if0_pl0_cd0.append(data[i+6,:])
    if0_pl0_cd1.append(data[i+7,:])
    if2_pl1_cd0.append(data[i+8,:])
    if2_pl1_cd1.append(data[i+9,:])
    if2_pl0_cd0.append(data[i+10,:])
    if2_pl0_cd1.append(data[i+11,:])
    if3_pl1_cd0.append(data[i+12,:])
    if3_pl1_cd1.append(data[i+13,:])
    if3_pl0_cd0.append(data[i+14,:])
    if3_pl0_cd1.append(data[i+15,:])

data_grouped = np.array([if1_pl1_cd0,if1_pl1_cd1,if1_pl0_cd0,if1_pl0_cd1,if0_pl1_cd0,if0_pl1_cd1,if0_pl0_cd0,if0_pl0_cd1,if2_pl1_cd0,if2_pl1_cd1,if2_pl0_cd0,if2_pl0_cd1,if3_pl1_cd0,if3_pl1_cd1,if3_pl0_cd0,if3_pl0_cd1])
print('Data shape: '+str(data_grouped.shape))


print('Saving data under '+outfile)
np.save(outfile, data_grouped)
print('Done')



