cmake_minimum_required(VERSION 2.6)
project(MITIGATE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")

find_package(Threads REQUIRED)
find_package(HTGS REQUIRED)
#find_package(FFTW REQUIRED)

# Add FFTW includes
# include_directories(${FFTW_INCLUDES})

# Add FFTW libs
# add_libraries(${FFTW_LIBRARIES})

#add_definitions(-DPROFILE)

include_directories(${HTGS_INCLUDE_DIR})
link_libraries(${CMAKE_THREAD_LIBS_INIT})

set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Wno-unused-parameter -Wno-reorder")
if(CMAKE_USE_PTHREADS_INIT)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -pthread")
endif(CMAKE_USE_PTHREADS_INIT)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        data/StreamData.h
        data/SpectrumData.h
        tasks/ReadStreamTask.cpp
        tasks/ReadStreamTask.h
        tasks/FFTTask.cpp
        tasks/FFTTask.h
        tasks/MitigateTask.cpp
        tasks/MitigateTask.h
        tasks/IFFTTask.cpp
        tasks/IFFTTask.h
        tasks/WriteResultTask.cpp
        tasks/WriteResultTask.h
        rules/GatherRule.cpp
        rules/GatherRule.h
        memory/DataBlockAllocator.h
        memory/SimpleReleaseRule.h)

add_executable(mitigate main.cpp ${SOURCE_FILES})