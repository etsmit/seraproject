//
// Created by tjb3 on 8/7/17.
//

#ifndef MITIGATE_DATABLOCKALLOCATOR_H
#define MITIGATE_DATABLOCKALLOCATOR_H
#include <htgs/api/IMemoryAllocator.hpp>
class DataBlockAllocator : public htgs::IMemoryAllocator<char> {
 public:
  DataBlockAllocator(size_t size) : IMemoryAllocator(size) {}

  char *memAlloc(size_t size) override {
    return (char *)malloc(size);
  }
  char *memAlloc() override {
    return (char *)malloc(this->size());
  }
  void memFree(char *&memory) override {
    free(memory);
  }

};
#endif //MITIGATE_DATABLOCKALLOCATOR_H
