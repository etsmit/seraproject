#INFO_rawvegas.py
#simply reads some info from a raw vegas file to help opening it

#second argument (boolean) is to show columns



import numpy as np
import astropy.io.fits as pyfits
import os,sys



#pulls from my scratch directory if full path not given
if sys.argv[1][0] != '/':
	inputFileName = my_dir + sys.argv[1]
else:
	inputFileName = sys.argv[1]

show_cols=sys.argv[2] #boolean


print('Reading Data...')
hdulist = pyfits.open(inputFileName)
datatable = hdulist['SINGLE DISH'].data
data = datatable.field('DATA')

if bool(show_cols):
	cols=hdulist['SINGLE DISH'].columns
	print(cols)


print('Data size: '+str(data.shape))

pols= datatable.field('PLNUM')
ifs = datatable.field('IFNUM')
cals= datatable.field('CAL')
fds= datatable.field('FDNUM')
sigs= datatable.field('SIG')
times = datatable.field('TIMESTAMP')

print('---------------------------------------')
print('Polarizations:')
print(pols[:30])

print('---------------------------------------')
print('Int. Freqs:')
print(ifs[:30])

print('---------------------------------------')
print('CalDiode States')
print(cals[:30])

print('---------------------------------------')
print('Feeds:')
print(pols[:30])

print('---------------------------------------')
print('SIG states:')
print(pols[:30])

print('---------------------------------------')
print('Time stamps:')
print(times[:30])

#if show_scans:
#	print('----------------------------------------')
#	print('Scan summary:')
#	scans=datatable.field('SCAN')
#	objs = datatable.field('OBJECT')
#	for i in range(len(scans)):
#		print(str(scans[i])+'   '+str(objs[i]))
