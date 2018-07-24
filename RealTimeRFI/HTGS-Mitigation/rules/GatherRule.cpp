//
// Created by tjb3 on 7/24/17.
//

#include "GatherRule.h"
GatherRule::GatherRule() : IRule() {

  // Allocate state container for storing each channel and time slice.
  // One way that this could be done is have a state container per channel and time point
  // Change as needed.
  //stateData = this->allocStateContainer(numChannels, numTimePoints);
  //fftCount = this->allocStateContainer<int>(numChannels, numTimePoints, 0);

}
GatherRule::~GatherRule() {
  // Release the memory
//  delete stateData;
//  delete fftCount;

}
std::string GatherRule::getName() {
  return "GatherRule";
}
void GatherRule::applyRule(std::shared_ptr<SpectrumData> data, size_t pipelineId) {
  // Store the data in the stateData (if needed), might be able to just keep track of the count.
  // Increment count per channel and time point

  // Add data to next task when all FFTs are done
  // if (countNeeded == fftCount->get(...))
  //   this->addResult(data)
}
