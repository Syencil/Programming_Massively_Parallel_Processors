cmake_minimum_required(VERSION 3.5)

cuda_add_executable(Convolutional Convolutional.cu)
set_property(TARGET Convolutional PROPERTY FOLDER PMPP)
target_link_libraries(Convolutional cudart.so  libopencv_core.so libopencv_imgproc.so libopencv_imgcodecs.so)

cuda_add_executable(Prefixsum Prefixsum.cu)
set_property(TARGET Prefixsum PROPERTY FOLDER PMPP)
target_link_libraries(Prefixsum cudart.so)

cuda_add_executable(Histogram Histogram.cu)
set_property(TARGET Histogram PROPERTY FOLDER PMPP)
target_link_libraries(Histogram cudart.so)

cuda_add_executable(SparseMatrix SparseMatrix.cu)
set_property(TARGET SparseMatrix PROPERTY FOLDER PMPP)
target_link_libraries(SparseMatrix cudart.so)

cuda_add_executable(MergeSort MergeSort.cu)
set_property(TARGET MergeSort PROPERTY FOLDER PMPP)
target_link_libraries(MergeSort cudart.so)

cuda_add_executable(GraphSearch GraphSearch.cu)
set_property(TARGET GraphSearch PROPERTY FOLDER PMPP)
target_link_libraries(GraphSearch cudart.so)