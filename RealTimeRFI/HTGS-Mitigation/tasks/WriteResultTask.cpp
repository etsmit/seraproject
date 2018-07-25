//
// Created by tjb3 on 7/19/17.
//

#include "WriteResultTask.h"
WriteResultTask::WriteResultTask(FILE *outputFilePointer, const char * headerCard, int headerLength,
                                 int BLOCSIZE, int OBSNCHAN, long long fileBytes) : ITask(),
                                  outputFilePointer(outputFilePointer), headerCard(headerCard), headerLength(headerLength),
                                  BLOCSIZE(BLOCSIZE), OBSNCHAN(OBSNCHAN), fileBytes(fileBytes) {
}
WriteResultTask::~WriteResultTask() {

}
void WriteResultTask::initialize() {

}
void WriteResultTask::executeTask(std::shared_ptr<SpectrumData> data)   {
//
//  int blockId = data->getBlockId();
//
//  // Find block id to check if all channels have been computed
//  auto itPos = channelMap.find(blockId);
//
//  if (itPos != channelMap.end())
//  {
//    // If the block Id is within the Channel map then get the number of channels currently processed
//    auto numChannels = (*itPos).second;
//
//    if (numChannels == OBSNCHAN-1) {
//      // We have received enough channels based on OBSNCHAN, so write the data block to output
//      // Optional can erase
//      //channelMap.erase(blockId);
//      if (blocksPassed > 0)
//      {
//        fwrite(data->getHeaderSection(), headerLength, 1, outputFilePointer);
//      }
//
//      currentBytesPassed += headerLength;
//
//      fwrite(data->getDataBlock()->get(), BLOCSIZE, 1, outputFilePointer);
//
//      currentBytesPassed += BLOCSIZE;
//
//      //Print Progress percentage every 10 blocks
//      if (blocksPassed%10 == 0)
//      {
//        std::cout << "Current Bytes Passed: " << static_cast<long double>(currentBytesPassed) / static_cast<long double>(fileBytes) * 100 << "%" << std::endl;
//      }
//
//      // Can release the memory . . .
//      this->releaseMemory(data->getDataBlock());
//      if (blocksPassed > 0)
//        free(data->getHeaderSection());
//
//      blocksPassed+=1;
//    }
//    else
//    {
//      // Have not received enough channels, so increment
//      (*itPos).second++;
//    }
//  } else
//  {
//    // Did not find blockId, so store it and mark 1 channel has been processed for the block id
//    channelMap.insert(std::pair<int, int>(blockId, 1));
//  }
}
void WriteResultTask::shutdown() {
  fclose(outputFilePointer);
}
std::string WriteResultTask::getName() {
  return "WriteResultTask";
}
WriteResultTask *WriteResultTask::copy() {
  return new WriteResultTask(outputFilePointer, headerCard, headerLength, BLOCSIZE, OBSNCHAN, fileBytes);
}
