//
// Created by tjb3 on 7/19/17.
//

#include "IFFTTask.h"
IFFTTask::IFFTTask(size_t numThreads) : ITask(numThreads) {

}
IFFTTask::~IFFTTask() {

}
void IFFTTask::initialize() {

}
void IFFTTask::executeTask(std::shared_ptr<SpectrumData> data) {
  // Compute inverse FFT (maybe in-place?)

  // addResult(data);
}
void IFFTTask::shutdown() {

}
std::string IFFTTask::getName() {
  return "IFFTTask";
}
IFFTTask *IFFTTask::copy() {
  return new IFFTTask(this->getNumThreads());
}
