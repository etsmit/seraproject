Program from Summer 2017 at GBO.

We are not using MAD, but it is still helpful to have a working C++ implementation to see the program layout.

timeMitigation.cc: Reads GUPPI RAW data from disk, parses header, and does time domain MAD
* Set up R/W Files
* Set up loop through file
* For first block, extract relevant header information
* Read data block to memory
* Time domain MAD RFI Mitigation
* Write out data
* Continue loop, close files when done
