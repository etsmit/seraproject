//
// Created by tjb3 on 7/19/17.
//

//#include <cstring>
#include <string.h>
#include "ReadStreamTask.h"
#include "../memory/SimpleReleaseRule.h"
// Current specifies 1 thread for reading stream and is a start task
ReadStreamTask::ReadStreamTask(std::string inputFileName) : ITask(1, true, false, 0), inputFileName(inputFileName) {

  inputFilePointer = fopen(inputFileName.c_str(), "r");
}

ReadStreamTask::~ReadStreamTask() {
  free(overlapSection);
}

void ReadStreamTask::initialize() {


}
void ReadStreamTask::executeTask(std::shared_ptr<htgs::VoidData> data) {
//  while(currentBytesPassed < fileBytes) {
//
//    char *headerSection = nullptr;
//
//    if (blocksPassed > 0)
//    {
//       headerSection = (char *) malloc(headerLength);
//
//      fread(headerSection, headerLength, 1, inputFilePointer);
//    }
//
//    currentBytesPassed += headerLength;
//
//    int NDIM = (BLOCSIZE/(OBSNCHAN*NPOL*(NBITS/8)));
//
//    if (blocksPassed == 0)
//    {
//      std::cout<<"NDIM: "<<NDIM<<std::endl;
//    }
//
//    htgs::m_data_t<char> dataBlock = this->getMemory<char>("DataBlock", new SimpleReleaseRule());
//    fread(dataBlock->get(), BLOCSIZE, 1, inputFilePointer);
//
//    // Produce data for each channel . . .
//    for (int CHANNEL = 0; CHANNEL < OBSNCHAN; CHANNEL++)
//    {
//      int channelOffset = CHANNEL * (BLOCSIZE/OBSNCHAN);
//      int currentOffset = channelOffset + overlapBytes;
//
//      this->addResult(new StreamData(dataBlock, currentOffset, blocksPassed, headerSection, NDIM, CHANNEL));
//    }
//
//
//    //Update bytes passed
//    currentBytesPassed += BLOCSIZE;
//
//    //Update blocks passed
//    blocksPassed += 1;
//
//  }
  // Data received will be 'nullptr' as this is a start task

  // if multiple threads can get thread ID and use that to determine which stream the thread is reading
  //this->getThreadID() (get latest version of HTGS for this function)
  // this->getOwnerTaskManager()->getThreadId() (alternatively do this)
  // Add code for reading in file/stream
  // addResult(new StreamData(...));
}
void ReadStreamTask::shutdown() {
  fclose(inputFilePointer);
}
std::string ReadStreamTask::getName() {
  return "ReadStreamTask";
}
ReadStreamTask *ReadStreamTask::copy() {
  return new ReadStreamTask(inputFileName);
}
int ReadStreamTask::getNPOL() const {
  return NPOL;
}
int ReadStreamTask::getOVERLAP() const {
  return OVERLAP;
}
long long int ReadStreamTask::getFileBytes() const {
  return fileBytes;
}
int ReadStreamTask::getOBSNCHAN() const {
  return OBSNCHAN;
}
int ReadStreamTask::getBLOCSIZE() const {
  return BLOCSIZE;
}
const char *ReadStreamTask::getHeaderCard() const {
  return headerCard;
}
int ReadStreamTask::getHeaderLength() const {
  return headerLength;
}
void ReadStreamTask::readHeaderCard(FILE *outputFilePointer) {
  fseek(inputFilePointer, 0, SEEK_END);
  fileBytes = ftell(inputFilePointer);
  std::cout << "File Size is: " << fileBytes << std::endl;
  fseek(inputFilePointer, 0, SEEK_SET); //Send the pointer back to the beginning of the file


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
      std::cout << "Found NPOL" << std::endl;
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

  NPOL = 4;
  //Print off variables
  std::cout<<"NPOL: "<< NPOL << std::endl;
  std::cout<<"OBSNCHAN: "<<OBSNCHAN<<std::endl;
  std::cout<<"NBITS: "<<NBITS<<std::endl;
  std::cout<<"BLOCSIZE: "<<BLOCSIZE<<std::endl;
  std::cout<<"OVERLAP: "<<OVERLAP<<std::endl;

  //Data block size is known
  //Allocate that much memory to our pointer


  //Byte overlap at the beginning of the channel
  overlapBytes = OVERLAP * NPOL;
  overlapSection = (char *) malloc(OVERLAP * NPOL);

  //Determine the headerLength variable in bytes
  //Memory allocate that many bytes to the headerSection pointer
  headerLength = cardCounter * cardLength;

}
