//
// Created by tjb3 on 7/19/17.
//

#include "FFTTask.h"

FFTTask::FFTTask(size_t numThreads) : ITask(numThreads) {}
FFTTask::~FFTTask() {}

void FFTTask::executeTask(std::shared_ptr<StreamData> data) {
  // Compute FFT from stream data memory (could use in-place?)
  // Or StreamData memory to memory allocated for SpectrumData
  // addResult(new SpectrumData(...));
}

FFTTask *FFTTask::copy() {
  return new FFTTask(this->getNumThreads());
}
void FFTTask::initialize() {

}
void FFTTask::shutdown() {
}
std::string FFTTask::getName() {
  return "FFTTask";
}
