#-----------------------------------------------
#abell370.py
#
#Takes data from a given abell370 vegas-fits file and puts it in npy form for easy SK analysis
#Inputs:
# 1: infile - raw file to open
# 2: outfile- npy filename to save under
#-----------------------------------------------

import numpy as np
import astropy.io.fits as pyfits
import os,sys
import matplotlib.pyplot as plt

infile = sys.argv[1]
outfile = sys.argv[2]


print('Reading Data...')
hdulist = pyfits.open(infile)
datatable = hdulist['SINGLE DISH'].data
data = datatable.field('DATA')

print('Data size: '+str(data.shape))

pols= datatable.field('PLNUM')
ifs = datatable.field('IFNUM')
cals= datatable.field('CAL')


#-----------------------------------------
#Each integration made up of 2 polarizations, 2 caldiode states and 4 ifs for 16 separate spectra

print('Reshaping data...')
#in the order they appear in the data
#if - intermediate frequency (0-3)
#pl - polarization (0 or 1)
#cd - caldiode state (0='F', 1='T')

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

for i in range(0,len(pols),16):
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

#put into a single array
data_grouped = np.array([if1_pl1_cd0,if1_pl1_cd1,if1_pl0_cd0,if1_pl0_cd1,if0_pl1_cd0,if0_pl1_cd1,if0_pl0_cd0,if0_pl0_cd1,if2_pl1_cd0,if2_pl1_cd1,if2_pl0_cd0,if2_pl0_cd1,if3_pl1_cd0,if3_pl1_cd1,if3_pl0_cd0,if3_pl0_cd1])

print('Regrouped into shape '+str(data_grouped.shape))
print('Saving data under '+str(outfile))
np.save(outfile,data_grouped)
print('Done!')



