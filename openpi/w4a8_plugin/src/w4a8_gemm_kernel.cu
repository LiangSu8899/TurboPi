/*
 * W4A8 GEMM CUDA Kernels
 *
 * Contains:
 * 1. FP8 dynamic quantization kernel (BF16 -> FP8 with per-row scaling)
 * 2. CUTLASS W4A8 GEMM wrapper (FP8 activation × NVFP4 weight -> BF16)
 *
 * Based on CUTLASS Example 72c for SM100 Block Scaled Tensor Core MMA.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>

// ============================================================================
// CUTLASS synclog workaround for host compilation
// These empty inline functions prevent compilation errors when CUTLASS
// __host__ __device__ functions try to call __device__-only synclog functions
// ============================================================================
namespace cutlass {
namespace arch {
  __host__ __device__ inline void synclog_emit_tma_store(int, uint64_t, uint32_t) {}
  __host__ __device__ inline void synclog_emit_tma_store_arrive(int) {}
  __host__ __device__ inline void synclog_emit_tma_store_wait(int, int) {}
  __host__ __device__ inline void synclog_emit_fence_view_async_shared(int) {}
  __host__ __device__ inline void synclog_emit_tma_load(int, uint64_t, uint32_t, uint32_t) {}
}
}

// CUTLASS includes
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ============================================================================
// Configuration
// ============================================================================

constexpr int FP8_E4M3_MAX = 448;  // FP8 E4M3 max representable value
constexpr int NVFP4_BLOCK_SIZE = 16;  // MX format block size
constexpr int WARP_SIZE = 32;

// ============================================================================
// FP8 Dynamic Quantization Kernel
// ============================================================================

__global__ void quantizeToFP8PerRowKernel(
    const __nv_bfloat16* __restrict__ input,  // [M, K]
    __nv_fp8_e4m3* __restrict__ output,       // [M, K]
    __nv_fp8_e4m3* __restrict__ scales,       // [M, num_blocks]
    int M, int K, int num_blocks
) {
    int row = blockIdx.x;
    if (row >= M) return;

    // Shared memory for reduction
    extern __shared__ float smem[];
    float* row_max = smem;

    // Step 1: Find per-block max (each warp handles one block)
    int block_idx = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int blocks_per_row = (K + NVFP4_BLOCK_SIZE - 1) / NVFP4_BLOCK_SIZE;

    if (block_idx < blocks_per_row) {
        float local_max = 0.0f;
        int block_start = block_idx * NVFP4_BLOCK_SIZE;

        // Each lane processes elements within the block
        for (int i = lane; i < NVFP4_BLOCK_SIZE && (block_start + i) < K; i += WARP_SIZE) {
            float val = __bfloat162float(input[row * K + block_start + i]);
            local_max = fmaxf(local_max, fabsf(val));
        }

        // Warp reduction to find block max
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
        }

        // Lane 0 stores the block max
        if (lane == 0) {
            row_max[block_idx] = local_max;
        }
    }
    __syncthreads();

    // Step 2: Quantize each block
    if (block_idx < blocks_per_row) {
        float block_max = row_max[block_idx];
        float scale = block_max / FP8_E4M3_MAX + 1e-12f;

        // Store scale (first lane only)
        if (lane == 0) {
            // Convert scale to FP8
            scales[row * num_blocks + block_idx].__x = __nv_cvt_float_to_fp8(scale, __NV_SATFINITE, __NV_E4M3);
        }

        int block_start = block_idx * NVFP4_BLOCK_SIZE;

        // Quantize elements
        for (int i = lane; i < NVFP4_BLOCK_SIZE && (block_start + i) < K; i += WARP_SIZE) {
            float val = __bfloat162float(input[row * K + block_start + i]);
            float scaled_val = val / scale;
            // Clamp and convert to FP8
            scaled_val = fminf(fmaxf(scaled_val, -FP8_E4M3_MAX), FP8_E4M3_MAX);
            output[row * K + block_start + i].__x = __nv_cvt_float_to_fp8(scaled_val, __NV_SATFINITE, __NV_E4M3);
        }
    }
}

extern "C" void quantizeActivationToFP8(
    const __nv_bfloat16* input,
    __nv_fp8_e4m3* output,
    __nv_fp8_e4m3* scales,
    int M, int K,
    cudaStream_t stream
) {
    int num_blocks = (K + NVFP4_BLOCK_SIZE - 1) / NVFP4_BLOCK_SIZE;
    int threads_per_row = num_blocks * WARP_SIZE;
    threads_per_row = min(threads_per_row, 1024);

    size_t smem_size = num_blocks * sizeof(float);

    quantizeToFP8PerRowKernel<<<M, threads_per_row, smem_size, stream>>>(
        input, output, scales, M, K, num_blocks
    );
}

// ============================================================================
// CUTLASS W4A8 GEMM Configuration
// ============================================================================

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// A matrix (activation): FP8 E4M3 with block scaling
using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;

// B matrix (weight): NVFP4 E2M1 with block scaling
using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128;

// Output: BF16
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Accumulator and architecture
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Tile configuration
using MmaTileShape = Shape<_256, _256, _256>;
using ClusterShape = Shape<_2, _4, _1>;

// Collective epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Collective mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// GEMM kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void
>;

using W4A8Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Type aliases for easy access
using StrideA = typename W4A8Gemm::GemmKernel::StrideA;
using StrideB = typename W4A8Gemm::GemmKernel::StrideB;
using StrideC = typename W4A8Gemm::GemmKernel::StrideC;
using StrideD = typename W4A8Gemm::GemmKernel::StrideD;
using LayoutSFA = typename W4A8Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename W4A8Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename W4A8Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

// ============================================================================
// W4A8 GEMM Forward
// ============================================================================

extern "C" void w4a8GemmForward(
    const void* activation,        // FP8 E4M3 [M, K]
    const void* activation_scale,  // FP8 scales [M, K/BLOCK_SIZE]
    const void* weight_packed,     // NVFP4 packed [K/2, N]
    const void* weight_scale,      // FP8 scales [K/BLOCK_SIZE, N]
    void* output,                  // BF16 [M, N]
    int M, int N, int K,
    float alpha,
    void* workspace,
    size_t workspaceSize,
    cudaStream_t stream
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    // Create strides
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    // Scale factor layouts
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    // Create GEMM instance
    W4A8Gemm gemm;

    // Arguments
    typename W4A8Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<const ElementA::DataType*>(activation), stride_A,
            reinterpret_cast<const ElementB::DataType*>(weight_packed), stride_B,
            reinterpret_cast<const ElementA::ScaleFactorType*>(activation_scale), layout_SFA,
            reinterpret_cast<const ElementB::ScaleFactorType*>(weight_scale), layout_SFB
        },
        {
            {alpha, 0.0f},
            nullptr, stride_C,  // No bias
            reinterpret_cast<ElementD*>(output), stride_D
        }
    };

    // Check if problem size is supported
    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        printf("W4A8 GEMM: Problem size not supported (M=%d, N=%d, K=%d)\n", M, N, K);
        return;
    }

    // Initialize and run
    status = gemm.initialize(arguments, workspace);
    if (status != cutlass::Status::kSuccess) {
        printf("W4A8 GEMM: Initialization failed\n");
        return;
    }

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        printf("W4A8 GEMM: Execution failed\n");
        return;
    }

#else
    printf("W4A8 GEMM: SM100 not supported on this device\n");
#endif
}

extern "C" size_t getW4A8GemmWorkspaceSize(int M, int N, int K) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    typename W4A8Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}
    };

    return W4A8Gemm::get_workspace_size(arguments);
#else
    return 0;
#endif
}

// ============================================================================
// Weight Quantization (for offline preprocessing)
// ============================================================================

__global__ void quantizeWeightToNVFP4Kernel(
    const __nv_bfloat16* __restrict__ input,  // [N, K] (transposed linear weight)
    uint8_t* __restrict__ output_packed,       // [K/2, N]
    __nv_fp8_e4m3* __restrict__ scales,        // [K/BLOCK_SIZE, N]
    int N, int K
) {
    // NVFP4 E2M1 values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
    const float NVFP4_MAX = 6.0f;
    const float NVFP4_VALUES[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

    int col = blockIdx.x;  // N dimension
    int block_idx = blockIdx.y;  // K blocks
    int block_start = block_idx * NVFP4_BLOCK_SIZE;

    if (col >= N || block_start >= K) return;

    // Step 1: Find block max
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < NVFP4_BLOCK_SIZE && (block_start + i) < K; i += blockDim.x) {
        float val = __bfloat162float(input[col * K + block_start + i]);
        local_max = fmaxf(local_max, fabsf(val));
    }

    // Reduce to find block max
    extern __shared__ float smem[];
    smem[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float block_max = smem[0];

    // Compute scale
    float scale = block_max / NVFP4_MAX + 1e-12f;

    // Store scale (thread 0 only)
    if (threadIdx.x == 0) {
        scales[block_idx * N + col].__x = __nv_cvt_float_to_fp8(scale, __NV_SATFINITE, __NV_E4M3);
    }

    // Step 2: Quantize to NVFP4 and pack
    for (int i = threadIdx.x * 2; i < NVFP4_BLOCK_SIZE && (block_start + i) < K; i += blockDim.x * 2) {
        uint8_t packed = 0;

        for (int j = 0; j < 2 && (block_start + i + j) < K; ++j) {
            float val = __bfloat162float(input[col * K + block_start + i + j]);
            float scaled = val / scale;

            // Find nearest NVFP4 value
            float sign = (scaled < 0) ? -1.0f : 1.0f;
            float abs_scaled = fabsf(scaled);

            int best_idx = 0;
            float best_dist = fabsf(abs_scaled - NVFP4_VALUES[0]);
            for (int k = 1; k < 8; ++k) {
                float dist = fabsf(abs_scaled - NVFP4_VALUES[k]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = k;
                }
            }

            // NVFP4 encoding: sign bit + 3-bit index
            uint8_t nvfp4_val = (sign < 0) ? (0x8 | best_idx) : best_idx;

            if (j == 0) {
                packed |= nvfp4_val;
            } else {
                packed |= (nvfp4_val << 4);
            }
        }

        // Store packed value
        int out_row = (block_start + i) / 2;
        output_packed[out_row * N + col] = packed;
    }
}

extern "C" void quantizeWeightToNVFP4(
    const __nv_bfloat16* input,
    uint8_t* output_packed,
    __nv_fp8_e4m3* scales,
    int N, int K,
    cudaStream_t stream
) {
    int num_k_blocks = (K + NVFP4_BLOCK_SIZE - 1) / NVFP4_BLOCK_SIZE;

    dim3 grid(N, num_k_blocks);
    dim3 block(min(NVFP4_BLOCK_SIZE, 32));
    size_t smem_size = block.x * sizeof(float);

    quantizeWeightToNVFP4Kernel<<<grid, block, smem_size, stream>>>(
        input, output_packed, scales, N, K
    );
}
