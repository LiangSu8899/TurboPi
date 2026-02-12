cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(W4A8GemmPlugin LANGUAGES CXX CUDA)

# ============================================================================
# Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# SM110 (Thor) architecture - Compute Capability 11.0
set(CMAKE_CUDA_ARCHITECTURES "110a")

# ============================================================================
# Find Dependencies
# ============================================================================

# CUDA
find_package(CUDAToolkit REQUIRED)

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include /usr/local/include
          /usr/include/x86_64-linux-gnu
          /usr/local/tensorrt/include
          $ENV{TENSORRT_ROOT}/include
)

find_library(TENSORRT_LIBRARY nvinfer
    HINTS /usr/lib /usr/local/lib
          /usr/lib/x86_64-linux-gnu
          /usr/local/tensorrt/lib
          $ENV{TENSORRT_ROOT}/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_ROOT.")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")

# CUTLASS (from external directory)
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../external/cutlass_nvfp4_build/include")
set(CUTLASS_UTIL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../external/cutlass_nvfp4_build/tools/util/include")
if(NOT EXISTS "${CUTLASS_DIR}")
    message(FATAL_ERROR "CUTLASS not found at ${CUTLASS_DIR}")
endif()
message(STATUS "CUTLASS include: ${CUTLASS_DIR}")
message(STATUS "CUTLASS util: ${CUTLASS_UTIL_DIR}")

# ============================================================================
# Plugin Library
# ============================================================================

add_library(w4a8_gemm_plugin SHARED
    src/w4a8_gemm_plugin.cpp
    src/w4a8_gemm_kernel.cu
)

target_include_directories(w4a8_gemm_plugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TENSORRT_INCLUDE_DIR}
    ${CUTLASS_DIR}
    ${CUTLASS_UTIL_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(w4a8_gemm_plugin
    ${TENSORRT_LIBRARY}
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(w4a8_gemm_plugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# CUDA compile options - SM110A enables TMA and block-scaled tensor core MMA
target_compile_options(w4a8_gemm_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -arch=sm_110a
        --expt-relaxed-constexpr
        --expt-extended-lambda
                -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1
        -DCUTLASS_ARCH_MMA_SM110_SUPPORTED=1
        -DCUTLASS_ARCH_MMA_SM110A_ENABLED=1
        -O3
    >
)

# Set output name
set_target_properties(w4a8_gemm_plugin PROPERTIES
    OUTPUT_NAME "w4a8_gemm_plugin"
    PREFIX "lib"
    SUFFIX ".so"
)

# ============================================================================
# Python Bindings (optional, requires pybind11)
# ============================================================================

find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 QUIET)

# Python bindings disabled for now (TODO: create python_bindings.cpp)
# if(pybind11_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/python_bindings.cpp")
#     pybind11_add_module(w4a8_gemm_plugin_py
#         src/python_bindings.cpp
#     )
#     target_link_libraries(w4a8_gemm_plugin_py PRIVATE w4a8_gemm_plugin)
#     message(STATUS "Python bindings will be built")
# else()
message(STATUS "Python bindings skipped (not implemented yet)")
# endif()

# ============================================================================
# Test Executable
# ============================================================================

add_executable(test_w4a8_gemm
    tests/test_w4a8_gemm.cu
    src/w4a8_gemm_kernel.cu
)

target_include_directories(test_w4a8_gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUTLASS_DIR}
    ${CUTLASS_UTIL_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(test_w4a8_gemm
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(test_w4a8_gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(test_w4a8_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -arch=sm_110a
        --expt-relaxed-constexpr
        --expt-extended-lambda
                -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1
        -DCUTLASS_ARCH_MMA_SM110_SUPPORTED=1
        -DCUTLASS_ARCH_MMA_SM110A_ENABLED=1
        -O3
    >
)

# ============================================================================
# Install
# ============================================================================

install(TARGETS w4a8_gemm_plugin
    LIBRARY DESTINATION lib
)

install(FILES src/w4a8_gemm_plugin.h
    DESTINATION include
)

message(STATUS "")
message(STATUS "W4A8 GEMM Plugin Configuration:")
message(STATUS "  CUDA Architecture: sm_110a (Blackwell Thor)")
message(STATUS "  TensorRT: ${TENSORRT_LIBRARY}")
message(STATUS "  CUTLASS: ${CUTLASS_DIR}")
message(STATUS "")
