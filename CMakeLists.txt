cmake_minimum_required(VERSION 3.5)
project(ParallOptimization)

set_property(GLOBAL PROPERTY USE_FOLDERS on)

# output
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")
message(STATUS "Project_binary_dir : ${PROJECT_BINARY_DIR}")
# c++ 11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# find opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
if(NOT OpenCV_LIBRARY_DIRS)
    set(OpenCV_LIBRARY_DIRS /usr/local/lib)
    message(WARING " Can not find opencv lib. It will use the default path => ${OpenCV_LIBRARY_DIRS}")
endif()
link_directories(${OpenCV_LIBRARY_DIRS})
message(STATUS "OpenCV_INCLUDE_DIRS => ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBRARY_DIRS => ${OpenCV_LIBRARY_DIRS}")

if(NOT OpenCV_FOUND)
    message(ERROR "OpenCV not found!")
endif(NOT OpenCV_FOUND)

# find cuda
find_package(CUDA)
find_package(CUDA REQUIRED)

include_directories(/usr/local/cuda/include)
if(NOT CUDA_LIBRARY_DIRS)
    set(CUDA_LIBRARY_DIRS /usr/local/cuda/lib64)
    message(WARING " Can not find CUDA lib. It will use the default path => ${CUDA_LIBRARY_DIRS}")
endif()
link_directories(${CUDA_LIBRARY_DIRS})
message(STATUS "CUDA_INCLUDE_DIRS : ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARY_DIRS : ${CUDA_LIBRARY_DIRS}")

if (CUDA_FOUND)
    add_subdirectory(PMPP)
else()
    message("CUDA not found!")
endif()