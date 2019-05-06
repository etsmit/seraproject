#--------------------------------------------------
#btlraw_to_npy
#opens BTLraw data files (ex. GBT18B_342) and turns them into npy files)
#follows Nick Joslyns code very closely
#www.github.com/NickJoslyn/BL-RAW/
#Takes two inputs:
#1: input filename (BL raw format) ** for now just put one arbitrary character - it gets rewritten
#2: npy file to save to !!(do NOT attach .npy on the end - i'll do that)!! **also input a character
#--------------------------------------------------

import numpy as np
import os,sys

#input variables
inputFileName = sys.argv[1]
outfile = sys.argv[2]

#first define file to open because I had to move data to a different directory
inputFileName = '/home/scratch/esmith/blc00_guppi_58373_19977_F01417+1651_0013.0000.raw'
#and destination to save to since I'm running this out of git directory
outfile = '/users/esmith/RFI_MIT/SK_OHmegamasers/npybtldata/blc00'
print('Opening file: '+inputFileName)

#init datalists
x_real=[]
x_imag=[]
y_real=[]
y_imag=[]

#--------------------------------------------------
#Begin Nick's magic code
#My small edits are marked with '#ETS'
#--------------------------------------------------
readIn = np.memmap(inputFileName, dtype = 'int8', mode = 'r')
fileBytes = os.path.getsize(inputFileName)


cardLength = 80     #BL Raw Information

#Ensure the same card is not read twice
lineCounter = 0     
currentBytesPassed = 0

# ETS ALL the blocks!
blockNumber = 0
for CHANNEL in range(64):
	print('---------------------')
	print('Channel '+str(CHANNEL+1)+' of 64')
  	x_real=[]
	x_imag=[]
	y_real=[]
	y_imag=[]

	while (blockNumber < 128):
		#print('---------------------')
		#print('Block: '+str(blockNumber+1))
		#print('---------------------')
   
		# Loop through header information ----------------------------------
		headerLoop = True
		lineCounter = 0     #Ensure the same card is not read twice
		while(headerLoop):
    
			cardString = ''

        #Get the ASCII value of the card and convert to char
			for index in range(cardLength):
            			cardString += chr(readIn[currentBytesPassed + index + lineCounter * cardLength])

        #ETS no need for this
        #print(cardString)

        #Identify the end of the header
        #If not the end, find other useful parameters from header
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

    # End Header Loop-------------------------------------------------------------------
        
    # Padding Bytes
		if (DIRECTIO != 0):
			DIRECTIO_offset = 512 - (cardLength*lineCounter)%512
		else:
			DIRECTIO_offset = 0

    # Skip Header
		headerOffset = cardLength * lineCounter + DIRECTIO_offset
		currentBytesPassed += headerOffset

    # Number of time samples per channel per block
		NDIM = int(BLOCSIZE/(OBSNCHAN*NPOL*(NBITS/8)))
    #print("\nNDIM:", NDIM)

    
    #Put data into an easily parsed array
		dataBuffer = readIn[currentBytesPassed:currentBytesPassed + BLOCSIZE].reshape(OBSNCHAN, NDIM, NPOL)
    
    #ETS save the data
    #may just put it in a single 1D array for each pol/type and then reshape it
    #for CHANNEL in range(OBSNCHAN):
		x_real.append(dataBuffer[CHANNEL,:,0])
		x_imag.append(dataBuffer[CHANNEL,:,1])
		y_real.append(dataBuffer[CHANNEL,:,2])
		y_imag.append(dataBuffer[CHANNEL,:,3])
            
            # At this point, you could run an FFT, periodogram, etc.
        #if CHANNEL==0:    # A simple print statement is sufficient to show it worked
            #print("Length of block: ",len(dataBuffer[CHANNEL,:,0]))

    
    # When parsing the entire file, keep track of where you are at
		currentBytesPassed += BLOCSIZE
		blockNumber += 1

#--------------------------------------------------
#END Nick's magic code
#--------------------------------------------------


	print('Data has been read -- now reshaping...')


#combine real and imag into 2 arrays
	x_real = np.array(x_real).flatten()
	x_imag = np.array(x_imag).flatten()
	y_real = np.array(y_real).flatten()
	y_imag = np.array(y_imag).flatten()
	print('Amount of samples: '+str(len(x_real)*4))


	np_x = np.zeros(len(x_real),dtype=complex)
	np_y = np.zeros(len(y_real),dtype=complex)

	np_x.real = x_real
	np_x.imag = x_imag
	np_y.real = y_real
	np_y.imag = y_imag

  #split the 64 channels back out
  #no need
  #np_x = np.reshape(np_x, (64,-1))
  #np_y = np.reshape(np_y, (64,-1))

	print('New shape: '+str(np_x.shape))

#now save the data
#two filenames for each polarization
	out_x = outfile+'_x_'+str(CHANNEL)+'.npy'
	out_y = outfile+'_y_'+str(CHANNEL)+'.npy'
	print('Saving data under '+out_x+' and '+out_y)
	np.save(out_x,np_x)
	np.save(out_y,np_y)
print('Done!')













