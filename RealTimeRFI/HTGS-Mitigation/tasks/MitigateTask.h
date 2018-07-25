//
// Created by tjb3 on 7/19/17.
//

#ifndef MITIGATE_MITIGATETASK_H
#define MITIGATE_MITIGATETASK_H

#include <htgs/api/ITask.hpp>
#include "../data/SpectrumData.h"
#include "../data/StreamData.h"

#define outlierDistance 3
#define normalDistributionScale 1.4826

class MitigateTask : public htgs::ITask<SpectrumData, SpectrumData> {
 public:
  MitigateTask(size_t numThreads, int OVERLAP, int NPOL);
  void initialize() override;
  void shutdown() override;
  std::string getName() override;
  void executeTask(std::shared_ptr<SpectrumData> data) override;
  MitigateTask *copy() override;
  ~MitigateTask() override;

 private:

  const static int samplesPerTransform = 4096;
  int OVERLAP;
  int NPOL;
  char tempXR[samplesPerTransform];
  char tempXI[samplesPerTransform];
  char tempYR[samplesPerTransform];
  char tempYI[samplesPerTransform];
};

#endif //MITIGATE_MITIGATETASK_H
