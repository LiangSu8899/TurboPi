cmake_minimum_required(VERSION 3.18)
project(w4a16_gemm_plugin LANGUAGES CXX CUDA)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 110 CACHE STRING "CUDA architectures")

# Find CUDA
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# CUTLASS paths - using pytorch's bundled CUTLASS
set(CUTLASS_DIR "/opt/pytorch/ao/third_party/cutlass/include")
set(CUTLASS_UTIL_DIR "/opt/pytorch/ao/third_party/cutlass/tools/util/include")
set(CUTLASS_EXAMPLES_DIR "/opt/pytorch/ao/third_party/cutlass/examples")

if(NOT EXISTS ${CUTLASS_DIR})
    message(FATAL_ERROR "CUTLASS include directory not found at ${CUTLASS_DIR}")
endif()

# Library
add_library(w4a16_gemm_plugin SHARED
    src/w4a16_gemm_kernel.cu
    src/w4a16_gemm_plugin.cpp
)

target_include_directories(w4a16_gemm_plugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUTLASS_DIR}
    ${CUTLASS_UTIL_DIR}
    ${CUTLASS_EXAMPLES_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(w4a16_gemm_plugin PRIVATE
    CUDA::cudart
    Threads::Threads
)

# CUDA flags
target_compile_options(w4a16_gemm_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --expt-relaxed-constexpr
        --extended-lambda
        -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1
        -Xcompiler=-fPIC
        -Xptxas=-v
    >
)

# Test executable
add_executable(test_w4a16_gemm
    tests/test_w4a16_gemm.cu
    src/w4a16_gemm_kernel.cu
)

target_include_directories(test_w4a16_gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUTLASS_DIR}
    ${CUTLASS_UTIL_DIR}
    ${CUTLASS_EXAMPLES_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(test_w4a16_gemm PRIVATE
    CUDA::cudart
    CUDA::cublas
    Threads::Threads
)

target_compile_options(test_w4a16_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --expt-relaxed-constexpr
        --extended-lambda
        -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1
    >
)
