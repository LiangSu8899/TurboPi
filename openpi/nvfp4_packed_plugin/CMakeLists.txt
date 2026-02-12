cmake_minimum_required(VERSION 3.18)
project(nvfp4_packed_plugin LANGUAGES CXX CUDA)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA Architecture - SM110 for Thor
set(CMAKE_CUDA_ARCHITECTURES 89 90 100 110)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find TensorRT
# Try to find TensorRT via environment variable or common paths
if(DEFINED ENV{TENSORRT_ROOT})
    set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
elseif(EXISTS "/usr/include/x86_64-linux-gnu/NvInfer.h")
    set(TENSORRT_ROOT "/usr")
elseif(EXISTS "/usr/local/tensorrt")
    set(TENSORRT_ROOT "/usr/local/tensorrt")
else()
    message(WARNING "TensorRT not found. Set TENSORRT_ROOT environment variable.")
endif()

if(TENSORRT_ROOT)
    find_path(TENSORRT_INCLUDE_DIR
        NAMES NvInfer.h
        PATHS ${TENSORRT_ROOT}/include
              /usr/include/x86_64-linux-gnu
              /usr/include/aarch64-linux-gnu
    )
    find_library(TENSORRT_LIBRARY
        NAMES nvinfer
        PATHS ${TENSORRT_ROOT}/lib
              /usr/lib/x86_64-linux-gnu
              /usr/lib/aarch64-linux-gnu
    )
    find_library(TENSORRT_PLUGIN_LIBRARY
        NAMES nvinfer_plugin
        PATHS ${TENSORRT_ROOT}/lib
              /usr/lib/x86_64-linux-gnu
              /usr/lib/aarch64-linux-gnu
    )
endif()

# Plugin source files
set(PLUGIN_SOURCES
    src/nvfp4_packed_kernel.cu
    src/nvfp4_packed_plugin.cpp
)

# Build shared library
add_library(nvfp4_packed_plugin SHARED ${PLUGIN_SOURCES})

target_include_directories(nvfp4_packed_plugin PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(nvfp4_packed_plugin PUBLIC
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    CUDA::cudart
)

# CUDA compilation flags
target_compile_options(nvfp4_packed_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -Xcompiler=-fPIC
    >
)

# Install
install(TARGETS nvfp4_packed_plugin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES src/nvfp4_packed_plugin.h
    DESTINATION include
)

# ============================================================================
# Test executable
# ============================================================================
option(BUILD_TESTS "Build test executable" ON)

if(BUILD_TESTS)
    add_executable(test_nvfp4_packed tests/test_nvfp4_packed.cu)

    target_include_directories(test_nvfp4_packed PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${TENSORRT_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(test_nvfp4_packed PRIVATE
        nvfp4_packed_plugin
        CUDA::cudart
    )

    target_compile_options(test_nvfp4_packed PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
        >
    )
endif()

# ============================================================================
# Standalone kernel benchmark (no TRT dependency)
# ============================================================================
add_executable(benchmark_nvfp4_kernel tests/benchmark_kernel.cu)

target_include_directories(benchmark_nvfp4_kernel PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(benchmark_nvfp4_kernel PRIVATE
    CUDA::cudart
)

target_compile_options(benchmark_nvfp4_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
    >
)

# Print configuration
message(STATUS "")
message(STATUS "=== NVFP4 Packed Plugin Configuration ===")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT Library: ${TENSORRT_LIBRARY}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "=========================================")
