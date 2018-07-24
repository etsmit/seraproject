//
// Created by tjb3 on 7/19/17.
//

#ifndef MITIGATE_WRITERESULTTASK_H
#define MITIGATE_WRITERESULTTASK_H

#include <htgs/api/ITask.hpp>
#include <htgs/api/VoidData.hpp>
#include "../data/SpectrumData.h"
#include "../data/StreamData.h"
class WriteResultTask : public htgs::ITask<SpectrumData, htgs::VoidData> {
 public:
  WriteResultTask(FILE *outputFilePointer, const char *headerCard, int headerLength, int BLOCSIZE, int OBSNCHAN, long long fileBytes);
  ~WriteResultTask() override;
  void initialize() override;
  void executeTask(std::shared_ptr<SpectrumData> data) override;
  void shutdown() override;
  std::string getName() override;
  WriteResultTask *copy() override;
 private:
  FILE *outputFilePointer;
  std::string outputFileName;

  const char *headerCard;
  int headerLength;

  int blocksPassed = 0;
  int BLOCSIZE;
  int OBSNCHAN;

  long long currentBytesPassed = 0;
  long long fileBytes;

  // Map to identify if the specified channel is ready to be written
  std::unordered_map<int, int> channelMap;

};

#endif //MITIGATE_WRITERESULTTASK_H
