//
// Created by tjb3 on 7/19/17.
//

#ifndef MITIGATE_READSTREAMTASK_H
#define MITIGATE_READSTREAMTASK_H

#include <htgs/api/ITask.hpp>
#include <htgs/api/VoidData.hpp>
#include "../data/StreamData.h"

#define cardLength 80


class ReadStreamTask : public htgs::ITask<htgs::VoidData, StreamData> {
 public:
  // Pass in location of file and other parameters
  ReadStreamTask(std::string inputFileName);
  ~ReadStreamTask() override;
  void initialize() override;
  void executeTask(std::shared_ptr<htgs::VoidData> data) override;
  void shutdown() override;
  std::string getName() override;
  ReadStreamTask *copy() override;

  int getNPOL() const;
  int getOVERLAP() const;
  long long int getFileBytes() const;
  int getOBSNCHAN() const;
  int getBLOCSIZE() const;
  const char *getHeaderCard() const;
  int getHeaderLength() const;

  void readHeaderCard(FILE *outputFilePointer);

 private:
  FILE *inputFilePointer;
  std::string inputFileName;
  long long currentBytesPassed = 0;
  long long fileBytes;

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

  //Initialize the variables for those pointers
  int headerLength;
  int overlapBytes;

  int cardCounter = 0; //keep track of how many cards for currentBytesPassed counter

  //Declare pointers that will point to malloc()s when the size is known
  char* overlapSection;

};

#endif //MITIGATE_READSTREAMTASK_H
