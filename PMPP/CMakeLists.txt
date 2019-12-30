cmake_minimum_required(VERSION 3.5)

add_executable(helloword helloword.cpp)
set_property(TARGET helloword PROPERTY FOLDER PMPP)
target_link_libraries(helloword cudart.so  libopencv_core.so libopencv_imgproc.so libopencv_imgcodecs.so)


cuda_add_executable(Convolutional Convolutional.cu)
set_property(TARGET Convolutional PROPERTY FOLDER PMPP)
target_link_libraries(Convolutional cudart.so  libopencv_core.so libopencv_imgproc.so libopencv_imgcodecs.so)

cuda_add_executable(Prefixsum Prefixsum.cu)
set_property(TARGET Prefixsum PROPERTY FOLDER PMPP)
target_link_libraries(Prefixsum cudart.so)

cuda_add_executable(Histogram Histogram.cu)
set_property(TARGET Histogram PROPERTY FOLDER PMPP)
target_link_libraries(Histogram cudart.so)