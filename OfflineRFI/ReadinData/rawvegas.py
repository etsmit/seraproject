#rawvegas.py
#to open a raw vegas file
#this is purposefully left incomplete for now
#follow the results of INFO_rawvegas.py and the method/framework of abell370.py to complete the program
#to open your specific shape of raw vegas file


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

#----------------------------------------------------



#insert code like abell370.py lines 40-73 here














#----------------------------------------------------


#put into a single array
data_grouped = np.array([if1_pl1_cd0,if1_pl1_cd1,if1_pl0_cd0,if1_pl0_cd1,if0_pl1_cd0,if0_pl1_cd1,if0_pl0_cd0,if0_pl0_cd1,if2_pl1_cd0,if2_pl1_cd1,if2_pl0_cd0,if2_pl0_cd1,if3_pl1_cd0,if3_pl1_cd1,if3_pl0_cd0,if3_pl0_cd1])

print('Regrouped into shape '+str(data_grouped.shape))
print('Saving data under '+str(outfile))
np.save(outfile,data_grouped)
print('Done!')




