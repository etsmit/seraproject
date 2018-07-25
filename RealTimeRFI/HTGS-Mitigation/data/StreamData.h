//
// Created by tjb3 on 7/19/17.
//

#ifndef MITIGATE_STREAMDATA_H
#define MITIGATE_STREAMDATA_H

#include <htgs/api/IData.hpp>
class StreamData : public htgs::IData
{
 public:
  StreamData(htgs::m_data_t<char> dataBlock, int offset, int blockId, char *headerSection, int NDIM, int channelId) : dataBlock(dataBlock), offset(offset),
                                                                              blockId(blockId), headerSection(headerSection),
                                                                              NDIM(NDIM), channelId(channelId)
  {}

  htgs::m_data_t<char> getDataBlock() const {
    return dataBlock;
  }
  int getOffset() const {
    return offset;
  }

  int getBlockId() const {
    return blockId;
  }
  char *getHeaderSection() const {
    return headerSection;
  }

  int getNDIM() const {
    return NDIM;
  }
  int getChannelId() const {
    return channelId;
  }
 private:
  htgs::m_data_t<char>dataBlock;
  int offset;
  int blockId;
  char *headerSection;
  int NDIM;
  int channelId;
};

#endif //MITIGATE_STREAMDATA_H
