//
// Created by tjb3 on 7/19/17.
//

#ifndef MITIGATE_FFTTASK_H
#define MITIGATE_FFTTASK_H

#include <htgs/api/ITask.hpp>
#include "../data/SpectrumData.h"
#include "../data/StreamData.h"
class FFTTask : public htgs::ITask<StreamData, SpectrumData> {
 public:
  void executeTask(std::shared_ptr<StreamData> data) override;
  FFTTask *copy() override;
  FFTTask(size_t numThreads);
  ~FFTTask() override;
  void initialize() override;
  void shutdown() override;
  std::string getName() override;
};

#endif //MITIGATE_FFTTASK_H
