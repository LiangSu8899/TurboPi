# nvFP4 GEMM TensorRT Plugin - Packed FP4 Version
#
# Uses real 4-bit packed format for 8x bandwidth savings.
# V4 Warp Reduce kernel: 0.36ms vs TRT FP8 0.53ms (1.46x faster)
#
# Build:
#   mkdir build && cd build
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES=110
#   make -j$(nproc)

cmake_minimum_required(VERSION 3.18)
project(nvfp4_packed_plugin LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture (SM110 for Jetson Thor)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 110)
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find TensorRT
if(NOT TENSORRT_ROOT)
    set(TENSORRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT installation path")
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64)

find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Set TENSORRT_ROOT to your TensorRT installation.")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")

# ============================================================================
# Packed FP4 Plugin (Best Performance - 1.46x faster than TRT FP8)
# ============================================================================
add_library(nvfp4_packed_plugin SHARED
    nvfp4_gemm_packed_launcher.cu
    nvfp4_gemm_packed_plugin.cpp
)

target_include_directories(nvfp4_packed_plugin PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(nvfp4_packed_plugin
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    CUDA::cudart
)

# CUDA flags for packed plugin
target_compile_options(nvfp4_packed_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --use_fast_math
        -Xcompiler=-fPIC
        -Xptxas=-v
    >
)

# Set output properties
set_target_properties(nvfp4_packed_plugin PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================================
# Legacy TVM Plugin (for reference/comparison)
# ============================================================================
add_library(nvfp4_tvm_plugin SHARED
    nvfp4_gemm_launcher.cu
    nvfp4_tvm_plugin.cpp
)

target_include_directories(nvfp4_tvm_plugin PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(nvfp4_tvm_plugin
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    CUDA::cudart
)

target_compile_options(nvfp4_tvm_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --use_fast_math
        -Xcompiler=-fPIC
    >
)

set_target_properties(nvfp4_tvm_plugin PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================================
# Install
# ============================================================================
install(TARGETS nvfp4_packed_plugin nvfp4_tvm_plugin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    nvfp4_gemm_packed_plugin.h
    nvfp4_gemm_tvm_plugin.h
    DESTINATION include
)

# ============================================================================
# Test Executables
# ============================================================================
option(BUILD_TESTS "Build test executables" OFF)
if(BUILD_TESTS)
    # Test packed kernel (standalone, no TensorRT)
    add_executable(test_nvfp4_packed
        nvfp4_gemm_packed.cu
    )
    target_link_libraries(test_nvfp4_packed
        CUDA::cudart
    )
    set_target_properties(test_nvfp4_packed PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )

    # Test TVM plugin
    add_executable(test_nvfp4_tvm
        test_nvfp4_tvm.cu
    )
    target_link_libraries(test_nvfp4_tvm
        nvfp4_tvm_plugin
        CUDA::cudart
    )
endif()

message(STATUS "nvFP4 Packed Plugin configuration:")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  TensorRT: ${TENSORRT_ROOT}")
message(STATUS "  Plugins: nvfp4_packed_plugin (RECOMMENDED), nvfp4_tvm_plugin (legacy)")
