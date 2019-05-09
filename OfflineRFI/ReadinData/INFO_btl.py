#INFO_btl.py
#simply reads the header of a BTL raw file data block fro basic file info


import sys,os
import numpy as np



#inputFileName = sys.argv[1]
inputFileName = '/home/scratch/esmith/blc00_guppi_58373_19977_F01417+1651_0013.0000.raw'

readIn = np.memmap(inputFileName, dtype = 'int8', mode = 'r')
fileBytes = os.path.getsize(inputFileName)


cardLength = 80     #BL Raw Information


lineCounter = 0     
currentBytesPassed = 0

blockNumber = 0
while (blockNumber < 1):

   
	# Loop through header information ----------------------------------
	headerLoop = True
	lineCounter = 0     #Ensure the same card is not read twice
	while(headerLoop):
    
		cardString = ''

        #Get the ASCII value of the card and convert to char
		for index in range(cardLength):
            		cardString += chr(readIn[currentBytesPassed + index + lineCounter * cardLength])

        #ETS no need for this
       		print(cardString)

		if (cardString[:3] == 'END'):   #reached end of header
			headerLoop = False

		elif(cardString[:8] == 'OBSNCHAN'): #Number of Channels
			OBSNCHAN = int(cardString[9:].strip()) #remove white spaces and convert to int
		elif(cardString[:4] == 'NPOL'):     #Number of Polarizations * 2
			NPOL = int(cardString[9:].strip())
		elif(cardString[:5] == 'NBITS'):    #Number of Bits per Data Sample
			NBITS = int(cardString[9:].strip())
		elif(cardString[:7] == 'OVERLAP'):  #Number of Time Samples that Overlap Between Blocks
			OVERLAP = int(cardString[9:].strip())
		elif(cardString[:8] == 'BLOCSIZE'): #Duration of Data Block in Bytes
			BLOCSIZE = int(cardString[9:].strip())
		elif(cardString[:8] == 'DIRECTIO'):
			DIRECTIO = int(cardString[9:].strip())
		elif(cardString[:7] == 'OBSFREQ'):
			OBSFREQ = float(cardString[9:].strip())
		elif(cardString[:7] == 'CHAN_BW'):
			CHAN_BW = float(cardString[9:].strip())
		elif(cardString[:5] == 'OBSBW'):
			OBSBW = float(cardString[9:].strip())
		elif(cardString[:4] == 'TBIN'):
			TBIN = float(cardString[9:].strip())

		lineCounter += 1    #Go to Next Card in Header
	blocknumber += 1

