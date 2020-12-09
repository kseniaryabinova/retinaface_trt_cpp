cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(retinaface LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
find_package(CUDA REQUIRED)
enable_language(CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

if(DEFINED TensorRT_DIR)
    include_directories("${TensorRT_DIR}/include")
    link_directories("${TensorRT_DIR}/lib")
endif(DEFINED TensorRT_DIR)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(retinaface_plugins SHARED
        ${CMAKE_SOURCE_DIR}/src/plugins/decode/DecodePlugin.cpp
        ${CMAKE_SOURCE_DIR}/src/plugins/nms/NMSPlugin.cpp
        ${CMAKE_SOURCE_DIR}/src/plugins/decode/decode.h
        ${CMAKE_SOURCE_DIR}/src/plugins/decode/decode.cu
        ${CMAKE_SOURCE_DIR}/src/plugins/nms/nms.h
        ${CMAKE_SOURCE_DIR}/src/plugins/nms/nms.cu
        ${CMAKE_SOURCE_DIR}/src/plugins/utils.h
        )

add_library(retinaface_lib SHARED
        ${CMAKE_SOURCE_DIR}/src/plugins/decode/decode.h
        ${CMAKE_SOURCE_DIR}/src/plugins/decode/decode.cu
        ${CMAKE_SOURCE_DIR}/src/plugins/nms/nms.h
        ${CMAKE_SOURCE_DIR}/src/plugins/nms/nms.cu
        ${CMAKE_SOURCE_DIR}/src/plugins/utils.h
        ${CMAKE_SOURCE_DIR}/src/plugins/decode/DecodePlugin.cpp
        ${CMAKE_SOURCE_DIR}/src/plugins/decode/DecodePlugin.h
        ${CMAKE_SOURCE_DIR}/src/plugins/nms/NMSPlugin.cpp
        ${CMAKE_SOURCE_DIR}/src/plugins/nms/NMSPlugin.h
        ${CMAKE_SOURCE_DIR}/src/engine.h
        ${CMAKE_SOURCE_DIR}/src/engine.cpp
        ${CMAKE_SOURCE_DIR}/src/calibrator.h
        )

set_target_properties(retinaface_plugins PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(retinaface_plugins PUBLIC nvinfer ${NVINFER_LIB} cuda ${CUDA_LIBRARIES})

set_target_properties(retinaface_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(retinaface_lib PUBLIC nvinfer nvonnxparser ${OPENCV_LIBS})

find_package(OpenCV REQUIRED)
add_executable(export ${CMAKE_SOURCE_DIR}/src/export.cpp)
target_link_libraries(export PRIVATE retinaface_lib ${OpenCV_LIBS} stdc++fs)

find_package(OpenCV REQUIRED)
add_executable(infer ${CMAKE_SOURCE_DIR}/src/infer.cpp)
target_link_libraries(infer PRIVATE retinaface_lib ${OpenCV_LIBS} ${CUDA_LIBRARIES})
