# W4A16 MLP TensorRT Plugin
#
# Build instructions:
# mkdir build && cd build
# cmake .. -DCMAKE_CUDA_ARCHITECTURES=110 -DTENSORRT_ROOT=/path/to/tensorrt
# make -j
#
# Author: Claude Code
# Date: 2026-02-10

cmake_minimum_required(VERSION 3.18)
project(w4a16_mlp_plugin CUDA CXX)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Default to SM110 (Thor)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 110)
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# TensorRT path
if(NOT DEFINED TENSORRT_ROOT)
    set(TENSORRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT installation path")
endif()

# TensorRT includes and libs
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")

# Check if TensorRT exists
if(NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInfer.h")
    message(WARNING "TensorRT not found at ${TENSORRT_ROOT}, trying system paths...")
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include /usr/local/include)
    find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib /usr/local/lib)
else()
    find_library(TENSORRT_LIBRARY nvinfer HINTS ${TENSORRT_LIB_DIR})
endif()

# Plugin library
add_library(w4a16_mlp_plugin SHARED
    w4a16_mlp_launcher.cu
    w4a16_mlp_plugin.cu
)

target_include_directories(w4a16_mlp_plugin PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(w4a16_mlp_plugin PRIVATE
    ${TENSORRT_LIBRARY}
    CUDA::cudart
)

# CUDA compile options
target_compile_options(w4a16_mlp_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
        --use_fast_math
    >
)

# Test executable (standalone kernel test, no TRT)
add_executable(test_w4a16_mlp test_w4a16_mlp.cu w4a16_mlp_launcher.cu)

target_include_directories(test_w4a16_mlp PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(test_w4a16_mlp PRIVATE
    CUDA::cudart
)

target_compile_options(test_w4a16_mlp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
        --use_fast_math
    >
)

# Install
install(TARGETS w4a16_mlp_plugin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES w4a16_mlp_plugin.h
    DESTINATION include/turbo_pi
)

message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TENSORRT_ROOT: ${TENSORRT_ROOT}")
message(STATUS "TENSORRT_INCLUDE_DIR: ${TENSORRT_INCLUDE_DIR}")
