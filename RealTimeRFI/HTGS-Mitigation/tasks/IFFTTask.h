//
// Created by tjb3 on 7/19/17.
//

#ifndef MITIGATE_IFFTTASK_H
#define MITIGATE_IFFTTASK_H

#include <htgs/api/ITask.hpp>
#include "../data/SpectrumData.h"
class IFFTTask : public htgs::ITask<SpectrumData, SpectrumData> {
 public:
  IFFTTask(size_t numThreads);
  ~IFFTTask() override;
  void initialize() override;
  void executeTask(std::shared_ptr<SpectrumData> data) override;
  void shutdown() override;
  std::string getName() override;
  IFFTTask *copy() override;
};

#endif //MITIGATE_IFFTTASK_H
