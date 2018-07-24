//
// Created by tjb3 on 7/24/17.
//

#ifndef MITIGATE_GATHERRULE_H
#define MITIGATE_GATHERRULE_H

#include <htgs/api/IRule.hpp>
#include "../data/SpectrumData.h"
class GatherRule : public htgs::IRule<SpectrumData, SpectrumData> {

 public:
  GatherRule();
  ~GatherRule() override;
  std::string getName() override;
  void applyRule(std::shared_ptr<SpectrumData> data, size_t pipelineId) override;

 private:
//  htgs::StateContainer<SpectrumData> *spectrumData;
//  htgs::StateContainer<int> *fftCount;
};

#endif //MITIGATE_GATHERRULE_H
