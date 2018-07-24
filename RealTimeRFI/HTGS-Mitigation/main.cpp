#include <iostream>
#include <htgs/api/TaskGraphConf.hpp>
#include <htgs/api/TaskGraphRuntime.hpp>
#include "tasks/ReadStreamTask.h"
#include "tasks/FFTTask.h"
#include "tasks/MitigateTask.h"
#include "tasks/IFFTTask.h"
#include "tasks/WriteResultTask.h"
#include "rules/GatherRule.h"
#include "memory/DataBlockAllocator.h"

int main() {

  std::string inputFileName = "./guppiHTGS.raw";
  std::string outputFileName = "testOfCPipeline.raw";

  FILE * outputFilePointer = fopen(outputFileName.c_str(), "wb");


  // initial parameters
  size_t numFFTThreads = 1;
  size_t numMitigateThreads = 40;

  size_t numIFFTThreads = 1;

  size_t numDataBlocks = 3;

  // ... Add additional params/options/flags

  // Create tasks
  ReadStreamTask *readTask = new ReadStreamTask(inputFileName);
  readTask->readHeaderCard(outputFilePointer);
  FFTTask *fftTask = new FFTTask(numFFTThreads);
  MitigateTask *madTask = new MitigateTask(numMitigateThreads, readTask->getOVERLAP(), readTask->getNPOL());
  IFFTTask *ifftTask = new IFFTTask(numIFFTThreads);
  WriteResultTask *writeResultTask = new WriteResultTask(outputFilePointer,
                                                         readTask->getHeaderCard(),
                                                         readTask->getHeaderLength(),
                                                         readTask->getBLOCSIZE(),
                                                         readTask->getOBSNCHAN(),
                                                         readTask->getFileBytes());

  htgs::Bookkeeper<SpectrumData> *bk = new htgs::Bookkeeper<SpectrumData>();

  GatherRule *rule = new GatherRule();



  // build HTGS graph
  htgs::TaskGraphConf<htgs::VoidData, htgs::VoidData> *graph = new htgs::TaskGraphConf<htgs::VoidData, htgs::VoidData>();

  graph->addEdge(readTask, fftTask);
  graph->addEdge(fftTask, bk);
  graph->addRuleEdge(bk, rule, madTask);
  graph->addEdge(madTask, ifftTask);
  graph->addEdge(ifftTask, writeResultTask);

  graph->addMemoryManagerEdge("DataBlock", readTask, new DataBlockAllocator((size_t)readTask->getBLOCSIZE()), numDataBlocks, htgs::MMType::Static);


  graph->writeDotToFile("Mitigate-Graph-Pre-Exec.dot", DOTGEN_FLAG_HIDE_MEM_EDGES | DOTGEN_FLAG_SHOW_IN_OUT_TYPES);

  // Execute the graph
  htgs::TaskGraphRuntime * runtime = new htgs::TaskGraphRuntime(graph);


  //Wall Clock Run Time
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  //CPU Run Time
  clock_t t1, t2;
  t1 = clock();
  runtime->executeAndWaitForRuntime();

  graph->writeDotToFile("Mitigate-Graph-Post-Exec.dot", DOTGEN_FLAG_HIDE_MEM_EDGES | DOTGEN_FLAG_SHOW_IN_OUT_TYPES);
  graph->writeDotToFile("Mitigate-Graph-Post-Exec-Show-Full-Threading.dot", DOTGEN_FLAG_SHOW_ALL_THREADING);

  delete runtime;

  std::cout << "Done." << std::endl;

  //Wall Time
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "Total Wall-Time: " << elapsed_seconds.count() << std::endl;

  //CPU Time
  t2 = clock();
  float diff = ((float)t2-(float)t1);
  float seconds = diff/ CLOCKS_PER_SEC;
  std::cout << "CPU Time: " << seconds << std::endl;

  std::cout << "Finished Mitigate" << std::endl;


  return 0;
}