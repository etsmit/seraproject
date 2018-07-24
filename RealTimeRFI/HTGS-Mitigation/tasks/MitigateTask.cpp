//
// Created by tjb3 on 7/19/17.
//

#include <cstring>
#include <random>
#include <algorithm>
#include "MitigateTask.h"


int findMedian(char data[], int length)
{
  char sortedData[length];
  std::memcpy(sortedData, data, length);
  std::sort(sortedData, &sortedData[length]);

  int median;

  if (length%2 == 0)
  {
    median = (sortedData[length/2 - 1] + sortedData[length/2])/2;
  }
  else
  {
    median = sortedData[length/2];
  }
  return median;
}

//Median Absolute Deviation
int findMITIGATE(char data[], int length)
{
  int median = findMedian(data, length);

  char absArray[length];
  for (int i = 0; i < length; i++)
  {
    absArray[i] = std::abs(data[i] - median);
  }

  int mad = findMedian(absArray, length);
  return mad;
}

char* replaceOutliers(char data[], int length)
{
  int median = findMedian(data, length);
  int mad = findMITIGATE(data, length);
  float rstd = mad*normalDistributionScale;

  //Robust standard deviation cannot be 0
  if (rstd == 0)
  {
    rstd = 0.01;
  }

  //Thresholds for Replacement
  float top = median + outlierDistance * rstd;
  float bottom = median - outlierDistance * rstd;

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(median, rstd);

  for (int i = 0; i < length; i++)
  {
    if ((data[i] < bottom) ||(data[i] > top))
    {
      data[i] = char(distribution(generator));
      distribution.reset();
    }
  }
  return data;
}

MitigateTask::MitigateTask(size_t numThreads, int OVERLAP, int NPOL) : ITask(numThreads), OVERLAP(OVERLAP), NPOL(NPOL) {}
MitigateTask::~MitigateTask() {}

void MitigateTask::executeTask(std::shared_ptr<SpectrumData> data) {
  // Compute average from spectrum data to build a local power spectrum array
  // Using power spectrum array normalize, compute median, and MITIGATE
  // Identify freq channels w.r.t. spectral outliers
  // replace SpectrumData input data real and imaginary values based on flagged freq channels
//
//  int currentOffset = data->getOffset();
//  char *dataBlock = data->getDataBlock()->get();
//  int NDIM = data->getNDIM();
//
////  std::cout << "Processing channel " << data->getChannelId() << " for block " << data->getBlockId() << std::endl;
//
//  //Time Domain MITIGATE (Perfect multiple)--------
//
//  for (int timeWindow = 0; timeWindow < ((NDIM-OVERLAP)/samplesPerTransform); timeWindow++)
//  {
//    //Data stream alternates Xreal, Ximag, Yreal, Yimag
//    for (int i = 0; i < samplesPerTransform; i++)
//    {
//      std::memcpy(&tempXR[i], &dataBlock[currentOffset + 4*i], 1);
//      std::memcpy(&tempXI[i], &dataBlock[currentOffset + 4*i + 1], 1);
//      std::memcpy(&tempYR[i], &dataBlock[currentOffset + 4*i + 2], 1);
//      std::memcpy(&tempYI[i], &dataBlock[currentOffset + 4*i + 3], 1);
//    }
//
//    char* newXR = replaceOutliers(tempXR, samplesPerTransform);
//    char* newXI = replaceOutliers(tempXI, samplesPerTransform);
//    char* newYR = replaceOutliers(tempYR, samplesPerTransform);
//    char* newYI = replaceOutliers(tempYI, samplesPerTransform);
//
//    //Put altered values back in original location
//    for (int i = 0; i < samplesPerTransform; i++)
//    {
//      std::memcpy(&dataBlock[currentOffset + 4*i], & newXR[i], 1);
//      std::memcpy(&dataBlock[currentOffset + 4*i + 1], &newXI[i], 1);
//      std::memcpy(&dataBlock[currentOffset + 4*i + 2], &newYR[i], 1);
//      std::memcpy(&dataBlock[currentOffset + 4*i + 3], &newYI[i], 1);
//    }
//
//    currentOffset += (samplesPerTransform * NPOL);
//
//  }
//
//  //Time Domain MITIGATE (Bytes leftover -- not caught by previous loop)----------
//  //Essentially a carbon copy of the above for loops
//  int samplesLeft = (NDIM - OVERLAP) - (samplesPerTransform)*((NDIM-OVERLAP)/samplesPerTransform);
//
//  if (samplesLeft > 0)
//  {
//    for (int i = 0; i < samplesLeft; i++)
//    {
//      std::memcpy(&tempXR[i], &dataBlock[currentOffset + 4*i], 1);
//      std::memcpy(&tempXI[i], &dataBlock[currentOffset + 4*i + 1], 1);
//      std::memcpy(&tempYR[i], &dataBlock[currentOffset + 4*i + 2], 1);
//      std::memcpy(&tempYI[i], &dataBlock[currentOffset + 4*i + 3], 1);
//    }
//
//    char* newXR = replaceOutliers(tempXR, samplesLeft);
//    char* newXI = replaceOutliers(tempXI, samplesLeft);
//    char* newYR = replaceOutliers(tempYR, samplesLeft);
//    char* newYI = replaceOutliers(tempYI, samplesLeft);
//
//    for (int i = 0; i < samplesLeft; i++)
//    {
//      std::memcpy(&dataBlock[currentOffset + 4*i], & newXR[i], 1);
//      std::memcpy(&dataBlock[currentOffset + 4*i + 1], &newXI[i], 1);
//      std::memcpy(&dataBlock[currentOffset + 4*i + 2], &newYR[i], 1);
//      std::memcpy(&dataBlock[currentOffset + 4*i + 3], &newYI[i], 1);
//    }
//
//    currentOffset += (samplesLeft * NPOL);
//  }

   addResult(data);
}

void MitigateTask::initialize() {

}
void MitigateTask::shutdown() {

}
std::string MitigateTask::getName() {
  return "MitigateTask";
}
MitigateTask *MitigateTask::copy() {
  return new MitigateTask(this->getNumThreads(), OVERLAP, NPOL);
}
