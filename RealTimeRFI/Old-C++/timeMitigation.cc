// Nick Joslyn and Emily Ramey
// 2017 National Radio Astronomy Observatory Summer Students
// Green Bank Observatory
// August 2, 2017
//---------------------------------------------------------------------------
// Program: timeMitigation.cc
// Purpose: RFI time domain mitigation on all channels of a full GUPPI file
// Notes: Intermediate Program en route to C++ pipeline -- not fully optimized
//---------------------------------------------------------------------------
#include <iostream>
#include <cstring>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <random>
#define cardLength 80 //80 characters is always the ASCII card length of each line in the header
#define outlierDistance 3 //3 standard deviations away
#define normalDistributionScale 1.4826 //MAD scale factor for Gaussian

//--------Functions-----------

int findMedian(char data[], int length)
{
	char sortedData[length];
	std::memcpy(sortedData, data, length);
	std::sort(sortedData, &sortedData[length]);

	int median;

	if (length%2 == 0)
	{
		median = (sortedData[length/2 - 1] + sortedData[length/2])/2;
	}
	else
	{
		median = sortedData[length/2];
	}
	return median;
}

//Median Absolute Deviation
int findMAD(char data[], int length)
{
	int median = findMedian(data, length);

	char absArray[length];
	for (int i = 0; i < length; i++)
	{
		absArray[i] = std::abs(data[i] - median);
	}

	int mad = findMedian(absArray, length);
	return mad;
}

char* replaceOutliers(char data[], int length)
{
	int median = findMedian(data, length);
	int mad = findMAD(data, length);
	float rstd = mad*normalDistributionScale;

	//Robust standard deviation cannot be 0
	if (rstd == 0)
	{
		rstd = 0.01;
	}

	//Thresholds for Replacement
	float top = median + outlierDistance * rstd;
	float bottom = median - outlierDistance * rstd;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(median, rstd);

	for (int i = 0; i < length; i++)
	{
		if ((data[i] < bottom) ||(data[i] > top))
		{
			data[i] = char(distribution(generator));
			distribution.reset();
		}
	}
	return data;
}

//----------------------------

int main() {

//-------Variables-----

	//File Pointers to keep track of location in file
	FILE *inputFilePointer;
	FILE *outputFilePointer;

	//Will hold user input for respective file names
	std::string inputFileName;
	std::string outputFileName;

	//Will determine when we break our while loop through the file
	long long currentBytesPassed = 0;
	long long fileBytes;

	//Number of time samples for each MAD window
	int samplesPerTransform = 4096;

//------Read/Write Files----------

	inputFileName = "guppi_56465_J1713+0747_0006.0000.raw";
	inputFilePointer = fopen(inputFileName.c_str(), "r");

	fseek(inputFilePointer, 0, SEEK_END);
	fileBytes = ftell(inputFilePointer);
	std::cout << "File Size is: " << fileBytes << std::endl;
	fseek(inputFilePointer, 0, SEEK_SET); //Send the pointer back to the beginning of the file

	outputFileName = "CTESTER.raw";
	outputFilePointer = fopen(outputFileName.c_str(), "wb");

	//Wall Clock Run Time
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	//CPU Run Time
	clock_t t1, t2;
	t1 = clock();

//-----Loop Through File---------

	//Counter for print outs
	int blocksPassed = 0;

	//Declare parameters whose value will be found on first iteration through header section
	int OBSNCHAN;
	int NPOL;
	int NBITS;
	int OVERLAP;
	int BLOCSIZE;

	//Arrays to store header ASCII data during the loop
	char headerCard[cardLength+1];
	char temp [cardLength+1];
	char temp2 [cardLength+1];

	//Declare pointers that will point to malloc()s when the size is known
	char* dataBlock;
	char* headerSection;
	char* overlapSection;

	//Initialize the variables for those pointers
	int headerLength;
	int overlapBytes;

	//Initialize variables for RFI time domain mitigation
	int overlapOffset;
	char tempXR[samplesPerTransform];
	char tempXI[samplesPerTransform];
	char tempYR[samplesPerTransform];
	char tempYI[samplesPerTransform];

	int cardCounter = 0; //keep track of how many cards for currentBytesPassed counter

	while (currentBytesPassed < fileBytes)
	{

		//===========Header information -- Only On First Iteration===============

		if (blocksPassed == 0)
		{
			while(true)
			{
				//Read a header card, append it to the output file
				fread(headerCard, cardLength, 1, inputFilePointer);
				headerCard[cardLength]='\n';
				fwrite(headerCard, 1, (sizeof(headerCard)-1), outputFilePointer);
				//------Find relevant parameters----------

				//Find the END keyword (77 space characters included so the strstr function doesn't find another END)
				if(strstr(headerCard, "END                                                                             "))
				{
					cardCounter += 1;
					break;	//End the loop because done with header section
				}

				else if(strstr(headerCard, "OBSNCHAN="))
				{
					//Identify the numerical part of the header card
					sscanf(headerCard, "%s %d", temp, &OBSNCHAN);
				}

				else if(strstr(headerCard, "NPOL    ="))
				{
					sscanf(headerCard, "%s %s %d", temp, temp2, &NPOL);
				}

				else if(strstr(headerCard, "NBITS   ="))
				{
					sscanf(headerCard, "%s %s %d", temp, temp2, &NBITS);
				}

				else if(strstr(headerCard, "OVERLAP ="))
				{
					sscanf(headerCard, "%s %s %d", temp, temp2, &OVERLAP);
				}

				else if(strstr(headerCard, "BLOCSIZE="))
				{
					sscanf(headerCard, "%s %d", temp, &BLOCSIZE);
				}

				cardCounter += 1;
			}

			//Print off variables
			std::cout<<"NPOL: "<< NPOL << std::endl;
			std::cout<<"OBSNCHAN: "<<OBSNCHAN<<std::endl;
			std::cout<<"NBITS: "<<NBITS<<std::endl;
			std::cout<<"BLOCSIZE: "<<BLOCSIZE<<std::endl;
			std::cout<<"OVERLAP: "<<OVERLAP<<std::endl;

			//Data block size is known
			//Allocate that much memory to our pointer
			dataBlock = (char *) malloc(BLOCSIZE);

			//Byte overlap at the beginning of the channel
			overlapBytes = OVERLAP * NPOL;
			overlapSection = (char *) malloc(OVERLAP * NPOL);

			//Determine the headerLength variable in bytes
			//Memory allocate that many bytes to the headerSection pointer
			headerLength = cardCounter * cardLength;
			headerSection = (char *) malloc(headerLength);
		}
		//===========End of Header Information Extraction============================

		//Read/write the header section if not the first block
		if (blocksPassed > 0)
		{
			fread(headerSection, headerLength, 1, inputFilePointer);
			fwrite(headerSection, headerLength, 1, outputFilePointer);
		}

		currentBytesPassed += headerLength;

		//Number of time samples per channel per block
		int NDIM = (BLOCSIZE/(OBSNCHAN*NPOL*(NBITS/8)));

		if (blocksPassed == 0)
		{
			std::cout<<"NDIM: "<<NDIM<<std::endl;
		}

		//One data block into memory
		fread(dataBlock, BLOCSIZE, 1, inputFilePointer);

		//=======================================================================
		//RFI Time Domain Mitigation

		for (int CHANNEL = 0; CHANNEL < OBSNCHAN; CHANNEL++)
		{
			int channelOffset = CHANNEL * (BLOCSIZE/OBSNCHAN);
			int currentOffset = channelOffset + overlapBytes;

			//Time Domain MAD (Perfect multiple)--------

			for (int timeWindow = 0; timeWindow < ((NDIM-OVERLAP)/samplesPerTransform); timeWindow++)
			{
				//Data stream alternates Xreal, Ximag, Yreal, Yimag
				for (int i = 0; i < samplesPerTransform; i++)
				{
					std::memcpy(&tempXR[i], &dataBlock[currentOffset + 4*i], 1);
					std::memcpy(&tempXI[i], &dataBlock[currentOffset + 4*i + 1], 1);
					std::memcpy(&tempYR[i], &dataBlock[currentOffset + 4*i + 2], 1);
					std::memcpy(&tempYI[i], &dataBlock[currentOffset + 4*i + 3], 1);
				}

				char* newXR = replaceOutliers(tempXR, samplesPerTransform);
				char* newXI = replaceOutliers(tempXI, samplesPerTransform);
				char* newYR = replaceOutliers(tempYR, samplesPerTransform);
				char* newYI = replaceOutliers(tempYI, samplesPerTransform);

				//Put altered values back in original location
				for (int i = 0; i < samplesPerTransform; i++)
				{
					std::memcpy(&dataBlock[currentOffset + 4*i], & newXR[i], 1);
					std::memcpy(&dataBlock[currentOffset + 4*i + 1], &newXI[i], 1);
					std::memcpy(&dataBlock[currentOffset + 4*i + 2], &newYR[i], 1);
					std::memcpy(&dataBlock[currentOffset + 4*i + 3], &newYI[i], 1);
				}

			currentOffset += (samplesPerTransform * NPOL);

			}

			//Time Domain MAD (Bytes leftover -- not caught by previous loop)----------
			//Essentially a carbon copy of the above for loops
			int samplesLeft = (NDIM - OVERLAP) - (samplesPerTransform)*((NDIM-OVERLAP)/samplesPerTransform);

			if (samplesLeft > 0)
			{
				for (int i = 0; i < samplesLeft; i++)
				{
					std::memcpy(&tempXR[i], &dataBlock[currentOffset + 4*i], 1);
					std::memcpy(&tempXI[i], &dataBlock[currentOffset + 4*i + 1], 1);
					std::memcpy(&tempYR[i], &dataBlock[currentOffset + 4*i + 2], 1);
					std::memcpy(&tempYI[i], &dataBlock[currentOffset + 4*i + 3], 1);
				}

				char* newXR = replaceOutliers(tempXR, samplesLeft);
				char* newXI = replaceOutliers(tempXI, samplesLeft);
				char* newYR = replaceOutliers(tempYR, samplesLeft);
				char* newYI = replaceOutliers(tempYI, samplesLeft);

				for (int i = 0; i < samplesLeft; i++)
				{
					std::memcpy(&dataBlock[currentOffset + 4*i], & newXR[i], 1);
					std::memcpy(&dataBlock[currentOffset + 4*i + 1], &newXI[i], 1);
					std::memcpy(&dataBlock[currentOffset + 4*i + 2], &newYR[i], 1);
					std::memcpy(&dataBlock[currentOffset + 4*i + 3], &newYI[i], 1);
				}

				currentOffset += (samplesLeft * NPOL);
			}
		}
		//=========================================================================

		//Write out section of data to the output file
		fwrite(dataBlock, BLOCSIZE, 1, outputFilePointer);

		//Update bytes passed
		currentBytesPassed += BLOCSIZE;

		//Print Progress percentage every 10 blocks
		if (blocksPassed%10 == 0)
		{
			std::cout << "Current Bytes Passed: " << static_cast<long double>(currentBytesPassed) / static_cast<long double>(fileBytes) * 100 << "%" << std::endl;
		}

		//Update blocks passed
		blocksPassed += 1;

	}

//------Clean and Finish Program----

	free(dataBlock);
	free(headerSection);
	fclose(inputFilePointer);
	fclose(outputFilePointer);

	std::cout << "Done." << std::endl;

	//Wall Time
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "Total Wall-Time: " << elapsed_seconds.count() << std::endl;

	//CPU Time
	t2 = clock();
	float diff = ((float)t2-(float)t1);
	float seconds = diff/ CLOCKS_PER_SEC;
	std::cout << "CPU Time: " << seconds << std::endl;

	return 0;

}
