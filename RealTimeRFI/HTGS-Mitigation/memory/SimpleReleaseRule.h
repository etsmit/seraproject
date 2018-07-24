//
// Created by tjb3 on 8/7/17.
//

#ifndef MITIGATE_SIMPLERELEASERULE_H
#define MITIGATE_SIMPLERELEASERULE_H

#include <htgs/api/IMemoryReleaseRule.hpp>
class SimpleReleaseRule : public htgs::IMemoryReleaseRule {
 public:
  void memoryUsed() override {

  }
  bool canReleaseMemory() override {
    return true;
  }
};

#endif //MITIGATE_SIMPLERELEASERULE_H
