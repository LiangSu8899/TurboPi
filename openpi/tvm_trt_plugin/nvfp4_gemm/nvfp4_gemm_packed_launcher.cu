/**
 * nvFP4 GEMM Packed Kernel Launcher
 *
 * Uses real 4-bit packed format (uint8, 2 FP4 per byte) instead of float32 simulation.
 * Based on the V4 Warp Reduce kernel that outperforms TRT FP8 by 1.46x.
 *
 * Memory comparison:
 * - float32 simulation: N*K*4 = 36 MB
 * - packed uint8:       N*K/2 = 4.5 MB (8x bandwidth savings!)
 *
 * Reference: TensorRT-LLM weightOnlyBatchedGemv
 * https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv
 *
 * Author: Claude Code
 * Date: 2026-02-10
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace turbo_pi {

// ============================================================================
// Constants
// ============================================================================
#define BLOCK_SIZE 32  // nvFP4 scaling block size
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// NVFP4 E2M1 decode table (4-bit -> float32)
// Encoding: sign(1) | magnitude_index(3)
// magnitude_index: 0->0, 1->0.5, 2->1, 3->1.5, 4->2, 5->3, 6->4, 7->6
__constant__ float NVFP4_DECODE_TABLE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,   // positive
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // negative
};

// ============================================================================
// Helper Functions
// ============================================================================
__device__ __forceinline__ float2 unpack_nvfp4_pair(uint8_t packed) {
    float2 result;
    result.x = NVFP4_DECODE_TABLE[packed & 0xF];
    result.y = NVFP4_DECODE_TABLE[(packed >> 4) & 0xF];
    return result;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// V4: Packed FP4 GEMV - Warp Reduce (Best Performance)
// Each warp (32 threads) computes one output element
// ============================================================================
__global__ void nvfp4_gemv_packed_v4_warp_reduce(
    const float* __restrict__ A,              // [1, K] activation (float32)
    const uint8_t* __restrict__ W_packed,     // [N, K/2] packed FP4 weights
    const float* __restrict__ scale_A,        // [1, num_blocks_k]
    const float* __restrict__ scale_W,        // [N, num_blocks_k]
    float* __restrict__ C,                    // [1, N] output
    int N, int K
) {
    // Each warp processes one output element
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_blocks_k = K / BLOCK_SIZE;

    if (warp_id < N) {
        int j = warp_id;
        float local_sum = 0.0f;

        // Each lane processes K/32 elements
        int k_per_lane = K / WARP_SIZE;
        int k_start = lane_id * k_per_lane;
        int k_end = k_start + k_per_lane;

        for (int k = k_start; k < k_end; k += 2) {
            int block_idx = k / BLOCK_SIZE;
            float a_scale = scale_A[block_idx];
            float w_scale = scale_W[j * num_blocks_k + block_idx];

            uint8_t w_packed = W_packed[j * (K / 2) + k / 2];
            float2 w_vals = unpack_nvfp4_pair(w_packed);

            float a0 = A[k] * a_scale;
            float a1 = A[k + 1] * a_scale;

            local_sum += a0 * w_vals.x * w_scale + a1 * w_vals.y * w_scale;
        }

        // Warp reduce
        float total = warpReduceSum(local_sum);

        // Lane 0 writes result
        if (lane_id == 0) {
            C[j] = total;
        }
    }
}

// ============================================================================
// V3: Vectorized Load (Alternative version for comparison)
// ============================================================================
__global__ void nvfp4_gemv_packed_v3_vectorized(
    const float* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_W,
    float* __restrict__ C,
    int N, int K
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks_k = K / BLOCK_SIZE;

    if (j < N) {
        float acc = 0.0f;

        // Process 8 elements at a time (4 bytes = 1 uint32)
        const uint32_t* W_packed_u32 = reinterpret_cast<const uint32_t*>(W_packed + j * (K / 2));

        for (int k = 0; k < K; k += 8) {
            int block_idx = k / BLOCK_SIZE;
            float a_scale = scale_A[block_idx];
            float w_scale = scale_W[j * num_blocks_k + block_idx];

            // Vectorized load: 4 bytes = 8 FP4 values
            uint32_t w_u32 = W_packed_u32[k / 8];

            // Unpack 8 FP4 values
            float w_vals[8];
            w_vals[0] = NVFP4_DECODE_TABLE[(w_u32 >> 0) & 0xF];
            w_vals[1] = NVFP4_DECODE_TABLE[(w_u32 >> 4) & 0xF];
            w_vals[2] = NVFP4_DECODE_TABLE[(w_u32 >> 8) & 0xF];
            w_vals[3] = NVFP4_DECODE_TABLE[(w_u32 >> 12) & 0xF];
            w_vals[4] = NVFP4_DECODE_TABLE[(w_u32 >> 16) & 0xF];
            w_vals[5] = NVFP4_DECODE_TABLE[(w_u32 >> 20) & 0xF];
            w_vals[6] = NVFP4_DECODE_TABLE[(w_u32 >> 24) & 0xF];
            w_vals[7] = NVFP4_DECODE_TABLE[(w_u32 >> 28) & 0xF];

            // Unrolled accumulation
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc += A[k + i] * a_scale * w_vals[i] * w_scale;
            }
        }

        C[j] = acc;
    }
}

// ============================================================================
// V1: Basic Packed (for reference/fallback)
// ============================================================================
__global__ void nvfp4_gemv_packed_v1_basic(
    const float* __restrict__ A,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_W,
    float* __restrict__ C,
    int N, int K
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks_k = K / BLOCK_SIZE;

    if (j < N) {
        float acc = 0.0f;

        for (int k = 0; k < K; k += 2) {
            int block_idx = k / BLOCK_SIZE;
            float a_scale = scale_A[block_idx];
            float w_scale = scale_W[j * num_blocks_k + block_idx];

            uint8_t w_packed = W_packed[j * (K / 2) + k / 2];
            float2 w_vals = unpack_nvfp4_pair(w_packed);

            float a0 = A[k] * a_scale;
            float a1 = A[k + 1] * a_scale;
            float w0 = w_vals.x * w_scale;
            float w1 = w_vals.y * w_scale;

            acc += a0 * w0 + a1 * w1;
        }

        C[j] = acc;
    }
}

// ============================================================================
// Launch Functions (exposed to TRT Plugin)
// ============================================================================

// Default: Use V4 Warp Reduce (best performance)
void launch_nvfp4_gemm_packed(
    const float* A,           // [M, K] activation
    const uint8_t* W_packed,  // [N, K/2] packed FP4 weights
    const float* scale_A,     // [M, num_blocks_k] activation scale
    const float* scale_W,     // [N, num_blocks_k] weight scale
    float* C,                 // [M, N] output
    int M, int N, int K,
    cudaStream_t stream
) {
    // For M=1 (single token inference), use warp reduce
    if (M == 1) {
        // V4: Each warp handles one output element
        int warps_per_block = THREADS_PER_BLOCK / WARP_SIZE;  // 8 warps/block
        int num_blocks = (N + warps_per_block - 1) / warps_per_block;

        nvfp4_gemv_packed_v4_warp_reduce<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            A, W_packed, scale_A, scale_W, C, N, K
        );
    } else {
        // For batched inference (M > 1), need to handle each row
        // TODO: Implement batched version
        // For now, process row by row
        for (int m = 0; m < M; ++m) {
            int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            nvfp4_gemv_packed_v1_basic<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                A + m * K, W_packed, scale_A + m * (K / BLOCK_SIZE), scale_W, C + m * N, N, K
            );
        }
    }
}

// Alternative: V3 Vectorized
void launch_nvfp4_gemm_packed_v3(
    const float* A,
    const uint8_t* W_packed,
    const float* scale_A,
    const float* scale_W,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    int num_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    nvfp4_gemv_packed_v3_vectorized<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        A, W_packed, scale_A, scale_W, C, N, K
    );
}

}  // namespace turbo_pi
