/**
 * N-Layer Persistent MLP Kernel for NVFP4 (Configurable)
 *
 * Based on GPT's recommendations:
 * - Start with 4-6 layers, sweep up to find sweet spot
 * - Verify with Nsight: no register spill, good occupancy
 * - Layer count is template parameter for optimal codegen
 *
 * Thor SM110 Constraints:
 * - L2 Cache: 128MB (effective ~51MB)
 * - FP4 weight per layer: ~60MB
 * - L2 can hold ~0.85 layers
 * - Register target: < 120 per thread
 * - Occupancy target: > 40%
 *
 * Author: Claude Code
 * Date: 2026-02-10
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

// ============================================================================
// Configurable Constants
// ============================================================================

// Model dimensions (PaLiGemma 2B)
constexpr int HIDDEN_SIZE = 2048;
constexpr int MLP_DIM = 16384;
constexpr int BLOCK_SIZE = 32;  // FP4 scaling block size

// Kernel config
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;

// Tile sizes - tuned for Thor
// TILE_MLP must be >= THREADS_PER_BLOCK so each thread has work
constexpr int TILE_MLP = 512;       // MLP elements per tile (2 per thread)
constexpr int TILE_K = 32;          // K elements per iteration (= BLOCK_SIZE)

// ============================================================================
// NVFP4 E2M1 Lookup Table
// ============================================================================

__constant__ float NVFP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float decode_fp4(uint8_t fp4_val) {
    return NVFP4_LUT[fp4_val & 0xF];
}

// ============================================================================
// Weight Structure (same for all layers)
// ============================================================================

struct LayerWeights {
    const uint8_t* gate_packed;   // [MLP_DIM, HIDDEN_SIZE/2]
    const float* gate_scale;      // [MLP_DIM, HIDDEN_SIZE/BLOCK_SIZE]
    const uint8_t* up_packed;     // [MLP_DIM, HIDDEN_SIZE/2]
    const float* up_scale;        // [MLP_DIM, HIDDEN_SIZE/BLOCK_SIZE]
    const uint8_t* down_packed;   // [HIDDEN_SIZE, MLP_DIM/2]
    const float* down_scale;      // [HIDDEN_SIZE, MLP_DIM/BLOCK_SIZE]
};

// ============================================================================
// Shared Memory Layout
// ============================================================================

// Double buffer for activation: 2 × 2048 × 4 = 16KB
// Weight decode buffer: 128 × 32 × 4 = 16KB
// Intermediate buffer: 128 × 4 = 512 bytes
// Total: ~33KB (fits in 48KB)

// ============================================================================
// N-Layer Persistent MLP Kernel (Template for layer count)
// ============================================================================

template<int NUM_LAYERS>
__global__ void nvfp4_nlayer_persistent_mlp_kernel(
    const float* __restrict__ input,        // [1, HIDDEN_SIZE]
    float* __restrict__ output,             // [1, HIDDEN_SIZE]
    const LayerWeights* __restrict__ layers // [NUM_LAYERS]
) {
    // Shared memory layout
    extern __shared__ char shared_mem[];
    float* smem_act_A = reinterpret_cast<float*>(shared_mem);
    float* smem_act_B = smem_act_A + HIDDEN_SIZE;
    float* smem_intermediate = smem_act_B + HIDDEN_SIZE;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Each thread handles HIDDEN_SIZE / THREADS_PER_BLOCK = 8 outputs
    constexpr int OUTPUTS_PER_THREAD = HIDDEN_SIZE / THREADS_PER_BLOCK;

    // ========================================================================
    // Step 1: Load input to shared memory (ONCE)
    // ========================================================================

    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD; i++) {
        int idx = tid * OUTPUTS_PER_THREAD + i;
        smem_act_A[idx] = input[idx];
    }
    __syncthreads();

    // Ping-pong buffers
    float* current_act = smem_act_A;
    float* next_act = smem_act_B;

    // ========================================================================
    // Step 2: Process NUM_LAYERS layers
    // ========================================================================

    #pragma unroll 1  // Don't unroll layer loop to save registers
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        const LayerWeights& w = layers[layer];

        const int num_blocks_hidden = HIDDEN_SIZE / BLOCK_SIZE;
        const int num_blocks_mlp = MLP_DIM / BLOCK_SIZE;

        // Zero output buffer
        #pragma unroll
        for (int i = 0; i < OUTPUTS_PER_THREAD; i++) {
            next_act[tid * OUTPUTS_PER_THREAD + i] = 0.0f;
        }
        __syncthreads();

        // Process MLP in tiles
        for (int mlp_tile = 0; mlp_tile < MLP_DIM; mlp_tile += TILE_MLP) {

            // Each thread computes a subset of MLP outputs for this tile
            constexpr int MLP_PER_THREAD = TILE_MLP / THREADS_PER_BLOCK;

            // Accumulators (in registers)
            float gate_acc[MLP_PER_THREAD];
            float up_acc[MLP_PER_THREAD];

            #pragma unroll
            for (int m = 0; m < MLP_PER_THREAD; m++) {
                gate_acc[m] = 0.0f;
                up_acc[m] = 0.0f;
            }

            // ================================================================
            // Gate + Up projection (fused)
            // ================================================================

            for (int k_block = 0; k_block < num_blocks_hidden; k_block++) {
                const int k_start = k_block * BLOCK_SIZE;

                // Load activation block from shared memory to registers
                float x_reg[BLOCK_SIZE / 4];  // Each thread loads subset
                const int x_per_thread = BLOCK_SIZE / 4;  // 8 elements

                #pragma unroll
                for (int i = 0; i < x_per_thread; i++) {
                    int k = k_start + (tid % 4) * x_per_thread + i;
                    if (k < HIDDEN_SIZE) {
                        x_reg[i] = current_act[k];
                    }
                }

                // Compute for each MLP output assigned to this thread
                #pragma unroll
                for (int m = 0; m < MLP_PER_THREAD; m++) {
                    const int m_idx = mlp_tile + tid * MLP_PER_THREAD + m;

                    if (m_idx < MLP_DIM) {
                        // Load scales
                        float gate_scale = w.gate_scale[m_idx * num_blocks_hidden + k_block];
                        float up_scale = w.up_scale[m_idx * num_blocks_hidden + k_block];

                        float local_gate = 0.0f;
                        float local_up = 0.0f;

                        // Process 2 elements at a time (1 byte = 2 FP4)
                        #pragma unroll
                        for (int k = 0; k < BLOCK_SIZE; k += 2) {
                            const int byte_idx = (k_start + k) / 2;

                            // Load packed FP4 weights
                            uint8_t gate_byte = w.gate_packed[m_idx * (HIDDEN_SIZE / 2) + byte_idx];
                            uint8_t up_byte = w.up_packed[m_idx * (HIDDEN_SIZE / 2) + byte_idx];

                            // Decode
                            float gate_low = decode_fp4(gate_byte & 0xF);
                            float gate_high = decode_fp4((gate_byte >> 4) & 0xF);
                            float up_low = decode_fp4(up_byte & 0xF);
                            float up_high = decode_fp4((up_byte >> 4) & 0xF);

                            // Get activation (from shared memory)
                            float x_low = current_act[k_start + k];
                            float x_high = current_act[k_start + k + 1];

                            // Accumulate
                            local_gate += gate_low * x_low + gate_high * x_high;
                            local_up += up_low * x_low + up_high * x_high;
                        }

                        // Apply scale
                        gate_acc[m] += local_gate * gate_scale;
                        up_acc[m] += local_up * up_scale;
                    }
                }
            }

            // ================================================================
            // SiLU activation and store intermediate
            // ================================================================

            #pragma unroll
            for (int m = 0; m < MLP_PER_THREAD; m++) {
                const int m_local = tid * MLP_PER_THREAD + m;

                // SiLU(gate) * up
                float gate = gate_acc[m];
                float sigmoid = 1.0f / (1.0f + expf(-gate));
                smem_intermediate[m_local] = gate * sigmoid * up_acc[m];
            }
            __syncthreads();

            // ================================================================
            // Down projection
            // ================================================================

            // Each thread contributes to its assigned hidden outputs
            #pragma unroll
            for (int h = 0; h < OUTPUTS_PER_THREAD; h++) {
                const int h_idx = tid * OUTPUTS_PER_THREAD + h;

                float h_acc = 0.0f;

                // Process the MLP tile
                const int mlp_blocks_in_tile = TILE_MLP / BLOCK_SIZE;

                for (int mb = 0; mb < mlp_blocks_in_tile; mb++) {
                    const int mlp_block_idx = (mlp_tile / BLOCK_SIZE) + mb;
                    float down_scale = w.down_scale[h_idx * num_blocks_mlp + mlp_block_idx];

                    float local_sum = 0.0f;

                    #pragma unroll
                    for (int m = 0; m < BLOCK_SIZE; m += 2) {
                        const int m_idx = mlp_tile + mb * BLOCK_SIZE + m;
                        const int m_local = mb * BLOCK_SIZE + m;
                        const int byte_idx = m_idx / 2;

                        // Load packed down weight
                        uint8_t down_byte = w.down_packed[h_idx * (MLP_DIM / 2) + byte_idx];

                        float down_low = decode_fp4(down_byte & 0xF);
                        float down_high = decode_fp4((down_byte >> 4) & 0xF);

                        // Get intermediate from shared memory
                        float inter_low = smem_intermediate[m_local];
                        float inter_high = smem_intermediate[m_local + 1];

                        local_sum += down_low * inter_low + down_high * inter_high;
                    }

                    h_acc += local_sum * down_scale;
                }

                // Atomic add (threads may write to same output)
                atomicAdd(&next_act[h_idx], h_acc);
            }
            __syncthreads();
        }

        // Swap buffers
        float* tmp = current_act;
        current_act = next_act;
        next_act = tmp;
        __syncthreads();
    }

    // ========================================================================
    // Step 3: Write output (ONCE)
    // ========================================================================

    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD; i++) {
        int idx = tid * OUTPUTS_PER_THREAD + i;
        output[idx] = current_act[idx];
    }
}

// ============================================================================
// Explicit Template Instantiations for Layer Sweep
// ============================================================================

template __global__ void nvfp4_nlayer_persistent_mlp_kernel<4>(
    const float*, float*, const LayerWeights*);

template __global__ void nvfp4_nlayer_persistent_mlp_kernel<6>(
    const float*, float*, const LayerWeights*);

template __global__ void nvfp4_nlayer_persistent_mlp_kernel<8>(
    const float*, float*, const LayerWeights*);

template __global__ void nvfp4_nlayer_persistent_mlp_kernel<10>(
    const float*, float*, const LayerWeights*);

template __global__ void nvfp4_nlayer_persistent_mlp_kernel<12>(
    const float*, float*, const LayerWeights*);

template __global__ void nvfp4_nlayer_persistent_mlp_kernel<18>(
    const float*, float*, const LayerWeights*);

// ============================================================================
// Host Interface
// ============================================================================

extern "C" {

// Calculate shared memory size
int get_smem_size() {
    // 2 activation buffers + intermediate buffer
    return 2 * HIDDEN_SIZE * sizeof(float) + TILE_MLP * sizeof(float);
}

// Launch kernels for different layer counts
void launch_4layer_persistent_mlp(
    const float* input, float* output,
    const LayerWeights* layers, cudaStream_t stream
) {
    int smem_size = get_smem_size();
    nvfp4_nlayer_persistent_mlp_kernel<4><<<1, THREADS_PER_BLOCK, smem_size, stream>>>(
        input, output, layers);
}

void launch_6layer_persistent_mlp(
    const float* input, float* output,
    const LayerWeights* layers, cudaStream_t stream
) {
    int smem_size = get_smem_size();
    nvfp4_nlayer_persistent_mlp_kernel<6><<<1, THREADS_PER_BLOCK, smem_size, stream>>>(
        input, output, layers);
}

void launch_8layer_persistent_mlp(
    const float* input, float* output,
    const LayerWeights* layers, cudaStream_t stream
) {
    int smem_size = get_smem_size();
    nvfp4_nlayer_persistent_mlp_kernel<8><<<1, THREADS_PER_BLOCK, smem_size, stream>>>(
        input, output, layers);
}

void launch_18layer_persistent_mlp(
    const float* input, float* output,
    const LayerWeights* layers, cudaStream_t stream
) {
    int smem_size = get_smem_size();
    nvfp4_nlayer_persistent_mlp_kernel<18><<<1, THREADS_PER_BLOCK, smem_size, stream>>>(
        input, output, layers);
}

// Get kernel attributes for profiling
void print_kernel_info() {
    cudaFuncAttributes attr;

    cudaFuncGetAttributes(&attr, nvfp4_nlayer_persistent_mlp_kernel<6>);
    printf("6-layer kernel:\n");
    printf("  Registers per thread: %d\n", attr.numRegs);
    printf("  Shared memory: %zu bytes\n", attr.sharedSizeBytes);
    printf("  Max threads per block: %d\n", attr.maxThreadsPerBlock);

    cudaFuncGetAttributes(&attr, nvfp4_nlayer_persistent_mlp_kernel<18>);
    printf("18-layer kernel:\n");
    printf("  Registers per thread: %d\n", attr.numRegs);
    printf("  Shared memory: %zu bytes\n", attr.sharedSizeBytes);
    printf("  Max threads per block: %d\n", attr.maxThreadsPerBlock);
}

}  // extern "C"
