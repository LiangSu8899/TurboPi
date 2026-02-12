/*
 * W4A16 GEMM CUTLASS Kernel Implementation
 *
 * Based on CUTLASS Example 86: blackwell_mixed_dtype_gemm
 * Optimized for SM110 (Jetson Thor)
 *
 * Key configuration:
 * - MmaType = bfloat16 (activation)
 * - QuantType = int4b_t (weight)
 * - MainloopSchedule = KernelTmaWarpSpecialized2SmMixedInputSm100
 *
 * Author: Claude Code
 * Date: 2026-02-09
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

// ============================================================================
// CUTLASS synclog workaround for host compilation
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
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ============================================================================
// CUTLASS Kernel Type Definitions
// ============================================================================

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Data types
using MmaType = cutlass::bfloat16_t;
using QuantType = cutlass::int4b_t;
using AccumulatorType = float;

// A matrix (activation): BF16, row-major
using ElementA = MmaType;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

// B matrix (weight): INT4, column-major
using ElementB = QuantType;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

// Transposed layouts for swap trick
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

// Scale and zero point types
using ElementZero = MmaType;
using ElementScale = MmaType;

// Output: BF16
using ElementC = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementD = cutlass::bfloat16_t;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Kernel configuration
using ElementAccumulator = AccumulatorType;
using ElementCompute = AccumulatorType;
using ArchTag = cutlass::arch::Sm100;  // Works on SM110 too
using OperatorClass = cutlass::arch::OpClassTensorOp;
using MmaTileShape = Shape<_256, _128, _128>;
using ClusterShape = Shape<_2, _1, _1>;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmMixedInputSm100;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

// Scale configuration
constexpr int ScaleGranularityN = 1;
constexpr int ScaleGranularityK = 128;
using ScaleConfig = cutlass::detail::Sm100MixedInputBlockwiseScaleConfig<ScaleGranularityN, ScaleGranularityK>;
using LayoutScale = decltype(ScaleConfig::deduce_layout_scale());

// Epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type, AlignmentC,
    ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD,
    EpilogueSchedule
>::CollectiveOp;

// Mainloop with scale only
using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementB, ElementScale>, cute::tuple<LayoutB_Transpose, LayoutScale>, AlignmentB,
    ElementA, LayoutA_Transpose, AlignmentA,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    MainloopSchedule
>::CollectiveOp;

using GemmKernelScaleOnly = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopScaleOnly,
    CollectiveEpilogue
>;

using GemmScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;

// Mainloop with scale and zero point
using CollectiveMainloopScaleWithZero = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementB, ElementScale, ElementZero>, cute::tuple<LayoutB_Transpose, LayoutScale>, AlignmentB,
    ElementA, LayoutA_Transpose, AlignmentA,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    MainloopSchedule
>::CollectiveOp;

using GemmKernelScaleWithZero = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloopScaleWithZero,
    CollectiveEpilogue
>;

using GemmScaleWithZero = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleWithZero>;

// Stride types
using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
using StrideC = typename GemmKernelScaleOnly::StrideC;
using StrideD = typename GemmKernelScaleOnly::StrideD;

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED || CUTLASS_ARCH_MMA_SM110_SUPPORTED

// ============================================================================
// Weight Quantization Kernels
// ============================================================================

// Per-group INT4 quantization kernel
__global__ void quantizeToINT4Kernel(
    const __nv_bfloat16* __restrict__ input,  // [N, K]
    int8_t* __restrict__ output_packed,        // [N, K/2]
    __nv_bfloat16* __restrict__ scales,        // [N, num_groups]
    __nv_bfloat16* __restrict__ zeros,         // [N, num_groups]
    int N, int K, int group_size
) {
    int n_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (n_idx >= N) return;

    int group_start = group_idx * group_size;
    int num_groups = (K + group_size - 1) / group_size;

    // Shared memory for reduction
    extern __shared__ float smem[];

    // Step 1: Find min and max in group
    float local_min = 1e10f;
    float local_max = -1e10f;

    for (int i = tid; i < group_size && (group_start + i) < K; i += blockDim.x) {
        float val = __bfloat162float(input[n_idx * K + group_start + i]);
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_min = fminf(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    // Block reduction
    if (tid % 32 == 0) {
        smem[tid / 32] = local_min;
        smem[tid / 32 + 32] = local_max;
    }
    __syncthreads();

    if (tid < 32) {
        local_min = (tid < blockDim.x / 32) ? smem[tid] : 1e10f;
        local_max = (tid < blockDim.x / 32) ? smem[tid + 32] : -1e10f;
        for (int offset = 16; offset > 0; offset /= 2) {
            local_min = fminf(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
    }
    __syncthreads();

    // Thread 0 computes scale and zero
    float group_min, group_max;
    if (tid == 0) {
        group_min = local_min;
        group_max = local_max;

        // Symmetric quantization: scale = max(|min|, |max|) / 7
        // INT4 range: [-8, 7], we use symmetric [-7, 7]
        float abs_max = fmaxf(fabsf(group_min), fabsf(group_max));
        float scale = abs_max / 7.0f + 1e-12f;
        float zero = 0.0f;  // Symmetric quantization

        scales[n_idx * num_groups + group_idx] = __float2bfloat16(scale);
        zeros[n_idx * num_groups + group_idx] = __float2bfloat16(zero);

        smem[0] = scale;
        smem[1] = zero;
    }
    __syncthreads();

    float scale = smem[0];
    float zero = smem[1];

    // Step 2: Quantize and pack
    // Process 2 elements at a time (pack into 1 byte)
    for (int i = tid * 2; i < group_size && (group_start + i) < K; i += blockDim.x * 2) {
        int8_t packed = 0;

        for (int j = 0; j < 2 && (group_start + i + j) < K; ++j) {
            float val = __bfloat162float(input[n_idx * K + group_start + i + j]);
            float scaled = (val - zero) / scale;
            // Clamp to INT4 range [-8, 7]
            int8_t q = static_cast<int8_t>(fmaxf(-8.0f, fminf(7.0f, roundf(scaled))));
            // Pack: low nibble first, then high nibble
            packed |= ((q & 0x0F) << (j * 4));
        }

        output_packed[n_idx * (K / 2) + (group_start + i) / 2] = packed;
    }
}

extern "C" void quantizeWeightToINT4(
    const __nv_bfloat16* input,
    int8_t* output_packed,
    __nv_bfloat16* scales,
    __nv_bfloat16* zeros,
    int N, int K,
    int group_size,
    cudaStream_t stream
) {
    int num_groups = (K + group_size - 1) / group_size;
    dim3 grid(N, num_groups);
    dim3 block(min(group_size, 256));
    size_t smem_size = 64 * sizeof(float);

    quantizeToINT4Kernel<<<grid, block, smem_size, stream>>>(
        input, output_packed, scales, zeros, N, K, group_size
    );
}

// Dequantization kernel for verification
__global__ void dequantizeINT4Kernel(
    const int8_t* __restrict__ input_packed,  // [N, K/2]
    const __nv_bfloat16* __restrict__ scales, // [N, num_groups]
    const __nv_bfloat16* __restrict__ zeros,  // [N, num_groups]
    __nv_bfloat16* __restrict__ output,       // [N, K]
    int N, int K, int group_size
) {
    int n_idx = blockIdx.x;
    int k_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (n_idx >= N || k_idx >= K) return;

    int num_groups = (K + group_size - 1) / group_size;
    int group_idx = k_idx / group_size;

    float scale = __bfloat162float(scales[n_idx * num_groups + group_idx]);
    float zero = __bfloat162float(zeros[n_idx * num_groups + group_idx]);

    // Unpack INT4
    int8_t packed = input_packed[n_idx * (K / 2) + k_idx / 2];
    int nibble = (k_idx % 2);
    int8_t q = (packed >> (nibble * 4)) & 0x0F;
    // Sign extend from 4-bit to 8-bit
    if (q & 0x08) q |= 0xF0;

    float val = static_cast<float>(q) * scale + zero;
    output[n_idx * K + k_idx] = __float2bfloat16(val);
}

extern "C" void dequantizeINT4ToBF16(
    const int8_t* input_packed,
    const __nv_bfloat16* scales,
    const __nv_bfloat16* zeros,
    __nv_bfloat16* output,
    int N, int K,
    int group_size,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(N, (K + block.x - 1) / block.x);

    dequantizeINT4Kernel<<<grid, block, 0, stream>>>(
        input_packed, scales, zeros, output, N, K, group_size
    );
}

// ============================================================================
// CUTLASS GEMM Forward Function
// ============================================================================

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Global storage for layouts (initialized per-call)
static LayoutScale g_layout_S;

template <typename Gemm, bool UseZeroPoint>
cudaError_t runGemm(
    const void* activation,
    const void* weight_packed,
    const void* weight_scale,
    const void* weight_zero,
    void* output,
    int M, int N, int K, int L,
    float alpha, float beta,
    void* workspace,
    size_t workspaceSize,
    cudaStream_t stream
) {
    // Compute strides
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(N, M, L));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(N, M, L));

    // Compute scale layout
    g_layout_S = ScaleConfig::tile_atom_to_shape_scale(make_shape(N, K, L));

    // Create GEMM arguments (A and B swapped)
    typename Gemm::Arguments arguments;

    if constexpr (UseZeroPoint) {
        arguments = typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {N, M, K, L},
            {reinterpret_cast<const ElementB*>(weight_packed), stride_B,
             reinterpret_cast<const ElementA*>(activation), stride_A,
             reinterpret_cast<const ElementScale*>(weight_scale), g_layout_S,
             reinterpret_cast<const ElementZero*>(weight_zero)},
            {{alpha, beta}, nullptr, stride_C,
             reinterpret_cast<ElementD*>(output), stride_D}
        };
    } else {
        arguments = typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {N, M, K, L},
            {reinterpret_cast<const ElementB*>(weight_packed), stride_B,
             reinterpret_cast<const ElementA*>(activation), stride_A,
             reinterpret_cast<const ElementScale*>(weight_scale), g_layout_S},
            {{alpha, beta}, nullptr, stride_C,
             reinterpret_cast<ElementD*>(output), stride_D}
        };
    }

    // Initialize GEMM
    Gemm gemm;
    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        printf("W4A16 GEMM: can_implement failed\n");
        return cudaErrorInvalidValue;
    }

    status = gemm.initialize(arguments, workspace);
    if (status != cutlass::Status::kSuccess) {
        printf("W4A16 GEMM: initialize failed\n");
        return cudaErrorInvalidValue;
    }

    // Run GEMM
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        printf("W4A16 GEMM: run failed\n");
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED || CUTLASS_ARCH_MMA_SM110_SUPPORTED

extern "C" void w4a16GemmForward(
    const void* activation,
    const void* weight_packed,
    const void* weight_scale,
    const void* weight_zero,
    void* output,
    int M, int N, int K,
    float alpha, float beta,
    void* workspace,
    size_t workspaceSize,
    cudaStream_t stream
) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    if (weight_zero != nullptr) {
        runGemm<GemmScaleWithZero, true>(
            activation, weight_packed, weight_scale, weight_zero,
            output, M, N, K, 1, alpha, beta,
            workspace, workspaceSize, stream
        );
    } else {
        runGemm<GemmScaleOnly, false>(
            activation, weight_packed, weight_scale, nullptr,
            output, M, N, K, 1, alpha, beta,
            workspace, workspaceSize, stream
        );
    }
#else
    printf("W4A16 GEMM requires SM100 or SM110 architecture\n");
#endif
}

extern "C" size_t getW4A16GemmWorkspaceSize(int M, int N, int K) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    // Compute strides
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(N, M, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(N, M, 1));

    g_layout_S = ScaleConfig::tile_atom_to_shape_scale(make_shape(N, K, 1));

    typename GemmScaleOnly::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {N, M, K, 1},
        {nullptr, stride_B, nullptr, stride_A, nullptr, g_layout_S},
        {{1.0f, 0.0f}, nullptr, stride_C, nullptr, stride_D}
    };

    return GemmScaleOnly::get_workspace_size(arguments);
#else
    return 0;
#endif
}
