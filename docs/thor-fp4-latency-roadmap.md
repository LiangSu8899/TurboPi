# Thor FP4 Latency-Winning Roadmap v2

> **核心策略调整**: 优先KV Projection + Cache Fuse，而非Multi-Layer Persistent MLP

---

## 为什么优先KV Fuse？

### ROI分析
| 优化目标 | 当前占比 | 潜在提升 | 整体收益 |
|----------|----------|----------|----------|
| **KV Pipeline** | **57%** | 50%↓ | **~1.4x 推理速度** |
| MLP | ~25% | 40%↓ | ~10% |

### 技术优势
1. **KV是纯memory-bound** → FP4最容易赢的地方
2. **GQA (num_kv_heads=1)** → KV weight被8个query共享，FP4 broadcast极其友好
3. **实现难度低** → 局部优化，易验证，对TRT plugin友好
4. **行业趋势** → FlashDecoding/PagedAttention/DeepSpeed全部首先攻KV

---

## Pi0.5 模型结构

### PaLiGemma (Gemma 2B)
| Parameter | Value | 对KV Fuse的意义 |
|-----------|-------|-----------------|
| hidden_size | 2048 | Q/K/V projection输入维度 |
| num_layers | 18 | 需要优化的层数 |
| num_heads | 8 | Query heads |
| **num_kv_heads** | **1 (GQA)** | **极高weight reuse** |
| head_dim | 256 | 每个head的维度 |

### Action Expert (Gemma 300M)
| Parameter | Value |
|-----------|-------|
| hidden_size | 1024 |
| num_layers | 18 |
| num_heads | 8 |
| **num_kv_heads** | **1 (GQA)** |
| head_dim | 256 |

---

## KV Fuse 核心优势

### 传统路径
```
load Wq weight (BF16)     → GEMV → Q
load Wk weight (BF16)     → GEMV → K → transpose → global store
load Wv weight (BF16)     → GEMV → V → transpose → global store
```
- 3次weight load
- 2次global store + transpose
- 无法fusion

### FP4 Fused 路径
```
load Wqkv packed (FP4)    → decode in smem (一次)
                          → compute Q, K, V (weight reuse)
                          → coalesced KV store (直接layout)
```
- Weight load: 3x → 1x (FP4更小)
- Decode overhead: amortized across Q/K/V
- Global write: 2x → 直接正确layout

### 预期收益
```
KV 57% → ~28%
整体: ~1.4x 推理速度
```

---

## 执行计划

### 🥇 Phase 1: KV Projection + Cache Write Fuse (Week 1-2)

#### 1.1 Shared Memory FP4 Decode Cache
```cuda
// 核心技术
__shared__ uint8_t w_packed_smem[TILE_K * TILE_N / 2];
__shared__ float w_scale_smem[TILE_N * (TILE_K / 32)];
__shared__ float w_decoded_smem[TILE_K * TILE_N];

// cp.async 异步加载
cp_async_cg(w_packed_smem, &W_qkv[tile_offset]);
cp_async_wait_group(0);

// 在 smem 中 decode
for (int i = tid; i < TILE_K * TILE_N / 2; i += blockDim.x) {
    uint8_t packed = w_packed_smem[i];
    int scale_idx = (i * 2) / 32;
    float scale = w_scale_smem[scale_idx];
    w_decoded_smem[i * 2] = NVFP4_DECODE[packed & 0xF] * scale;
    w_decoded_smem[i * 2 + 1] = NVFP4_DECODE[packed >> 4] * scale;
}
__syncthreads();

// 计算时复用 decoded weights
```

#### 1.2 Fused QKV Projection Kernel
```cuda
// Grid: (num_blocks_n, batch * seq, num_layers)
// Block: 256 threads

template<int HIDDEN_SIZE, int HEAD_DIM, int NUM_HEADS, int NUM_KV_HEADS>
__global__ void fused_qkv_fp4_kernel(
    const half* __restrict__ x,           // [B, S, hidden]
    const uint8_t* __restrict__ Wq,       // [num_heads * head_dim, hidden/2]
    const uint8_t* __restrict__ Wk,       // [num_kv_heads * head_dim, hidden/2]
    const uint8_t* __restrict__ Wv,       // [num_kv_heads * head_dim, hidden/2]
    const half* __restrict__ scale_q,
    const half* __restrict__ scale_k,
    const half* __restrict__ scale_v,
    half* __restrict__ Q_out,             // [B, S, num_heads, head_dim]
    half* __restrict__ K_cache,           // [B, max_seq, num_kv_heads, head_dim]
    half* __restrict__ V_cache,           // [B, max_seq, num_kv_heads, head_dim]
    int batch_size, int seq_len, int cache_pos
) {
    // 1. Load x to shared memory
    __shared__ half x_smem[HIDDEN_SIZE];
    cooperative_load(x, x_smem, blockIdx.y);

    // 2. For each output head dimension tile
    int head_idx = blockIdx.x / (HEAD_DIM / TILE_N);
    int tile_n = blockIdx.x % (HEAD_DIM / TILE_N);

    // 3. Decode FP4 weights in shared memory
    __shared__ half wq_decoded[TILE_K][TILE_N];
    __shared__ half wk_decoded[TILE_K][TILE_N];  // Reuse for GQA
    __shared__ half wv_decoded[TILE_K][TILE_N];

    // 4. Compute Q (for all 8 heads using different weight tiles)
    // 5. Compute K, V (for 1 KV head, shared across all Q heads)

    // 6. Direct write to KV cache with correct layout
    // 避免 transpose!
    if (threadIdx.x < TILE_N) {
        int kv_idx = head_idx / (NUM_HEADS / NUM_KV_HEADS);  // 0 for GQA
        K_cache[batch_idx * max_seq * NUM_KV_HEADS * HEAD_DIM +
                cache_pos * NUM_KV_HEADS * HEAD_DIM +
                kv_idx * HEAD_DIM +
                tile_n * TILE_N + threadIdx.x] = k_result[threadIdx.x];
        // V similar
    }
}
```

#### 1.3 Warp-Cooperative KV Write
```cuda
// 直接写入正确的 KV cache layout: [B, max_seq, num_kv_heads, head_dim]
// 避免中间 transpose

// Warp 0-3: 计算 Q heads 0-3
// Warp 4-7: 计算 Q heads 4-7
// Warp 0: 同时计算 K, V (因为 GQA 只有 1 个 KV head)

// Coalesced write pattern:
// 32 threads 写 32 个连续的 head_dim 元素
```

#### 1.4 验证实验
```python
# 对比
# A: Separate Q/K/V projections + KV cache write (cuBLAS)
# B: Fused QKV FP4 projection with direct KV cache write

def test_qkv_kv_fuse():
    # PaLiGemma dimensions
    hidden_size = 2048
    num_heads = 8
    num_kv_heads = 1
    head_dim = 256
    seq_len = 455  # 实际 prefix pass

    # Benchmark both paths
    baseline_time = benchmark_separate_qkv(...)
    fused_time = benchmark_fused_qkv_fp4(...)

    print(f"Baseline: {baseline_time:.3f} ms")
    print(f"Fused FP4: {fused_time:.3f} ms")
    print(f"Speedup: {baseline_time / fused_time:.2f}x")
```

**预期结果**: 延迟降低 > 50% (KV pipeline部分)

---

### 🥈 Phase 2: KV Cache Persistent Kernel (Week 2-3)

#### 2.1 SM Resident KV Shard
```cuda
// 保持 KV cache shard 在 SM shared memory
// 避免反复 global memory round-trip

// Grid: persistent (num_SMs blocks)
// Each block holds a shard of KV cache

__global__ void persistent_kv_kernel(...) {
    // Shared memory: 48KB per block
    // 可以 hold: 48KB / (256 * 2 * 2) = ~48 tokens per head
    __shared__ half kv_shard[48][256][2];  // [tokens, head_dim, k/v]

    while (has_work()) {
        // 1. Receive new token embedding
        // 2. Compute K, V projection (FP4)
        // 3. Update local KV shard
        // 4. Attention compute with local shard
        // 5. Only spill to global when shard full
    }
}
```

#### 2.2 Attention + KV Fuse
```cuda
// 进一步融合 attention 计算
// 避免 KV 写出再读回

fused_attention_with_kv_update(
    x,           // 当前 token embedding
    Wq, Wk, Wv,  // FP4 weights
    kv_cache,    // 只有 cache miss 时才访问
    output
) {
    // 1. Compute Q, K, V from x
    // 2. Update KV cache (如果需要)
    // 3. Attention: softmax(Q @ K^T) @ V
    // 4. Output projection
    // 全部在一个 kernel 里完成
}
```

---

### 🥉 Phase 3: Multi-Layer Persistent MLP (Week 3-4)

> 只有在 Phase 1, 2 成功后才值得做

#### 3.1 Persistent MLP Kernel
```cuda
// 合并多层 MLP 到一个 persistent kernel
// 减少 kernel launch overhead 和 L2 thrashing

__global__ void persistent_mlp_kernel(
    const half* x,
    const uint8_t* gate_weights[NUM_LAYERS],
    const uint8_t* up_weights[NUM_LAYERS],
    const uint8_t* down_weights[NUM_LAYERS],
    // scales...
    half* output,
    int num_layers
) {
    // Shared memory for activation caching
    __shared__ half x_smem[HIDDEN_SIZE];
    __shared__ half intermediate_smem[INTERMEDIATE_SIZE];

    for (int layer = 0; layer < num_layers; layer++) {
        // 1. gate = silu(x @ gate_weight[layer])
        // 2. up = x @ up_weight[layer]
        // 3. x = (gate * up) @ down_weight[layer]
        // 全部在 smem 中完成
    }
}
```

---

### 🏁 Phase 4: TVM/TRT Integration (Week 4-5)

#### 4.1 TVM TensorIR Schedule
```python
@T.prim_func
def fused_qkv_kv_cache_fp4(
    x: T.Buffer[(B, S, 2048), "float16"],
    Wq_packed: T.Buffer[(2048, 1024), "uint8"],
    Wk_packed: T.Buffer[(256, 1024), "uint8"],
    Wv_packed: T.Buffer[(256, 1024), "uint8"],
    scale_q: T.Buffer[(2048, 64), "float16"],
    scale_k: T.Buffer[(256, 64), "float16"],
    scale_v: T.Buffer[(256, 64), "float16"],
    Q_out: T.Buffer[(B, S, 8, 256), "float16"],
    K_cache: T.Buffer[(B, MAX_SEQ, 1, 256), "float16"],
    V_cache: T.Buffer[(B, MAX_SEQ, 1, 256), "float16"],
    cache_pos: T.int32,
):
    # Tile 策略
    for bx in T.thread_binding(NUM_BLOCKS, "blockIdx.x"):
        for tx in T.thread_binding(256, "threadIdx.x"):
            # Shared memory decode
            with T.block("decode"):
                # cp.async load FP4 weights
                # decode in smem
                pass

            # Compute QKV
            with T.block("qkv_gemv"):
                # GEMV with decoded weights
                pass

            # Direct KV cache write
            with T.block("kv_write"):
                # Coalesced write to correct layout
                pass
```

#### 4.2 TRT Plugin 封装
```cpp
class FusedQKVKVCacheFP4Plugin : public IPluginV2DynamicExt {
public:
    // 导出为 TRT plugin
    // 集成到现有 TRT pipeline

    int enqueue(
        const PluginTensorDesc* inputDesc,
        const PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream
    ) override {
        // Launch fused_qkv_kv_cache_fp4 kernel
        fused_qkv_kv_cache_fp4_kernel<<<grid, block, smem, stream>>>(
            inputs[0],  // x
            inputs[1],  // Wq_packed
            // ...
            outputs[0], // Q_out
            outputs[1], // K_cache (updated in-place)
            outputs[2], // V_cache (updated in-place)
            cache_pos
        );
        return 0;
    }
};
```

---

## Kernel 设计细节

### Grid Mapping (Phase 1)
```
Grid:  (num_head_tiles, batch * seq, 1)
Block: (256, 1, 1)

num_head_tiles = num_heads * (head_dim / TILE_N)
              = 8 * (256 / 32) = 64 for Q
              = 1 * (256 / 32) = 8  for K, V

Total blocks per token: 64 + 8 + 8 = 80
```

### Warp Role Split
```
Block = 256 threads = 8 warps

Warp 0-5: Q projection (6 warps, 8 heads, ~1.33 heads/warp)
Warp 6:   K projection (1 warp, 1 KV head)
Warp 7:   V projection (1 warp, 1 KV head)

或者更高效的 split:
Warp 0-7: 交替处理 Q/K/V tiles
          每个 warp 处理不同的 output tile
          共享 decoded weight in smem
```

### Shared Memory Layout
```
Total: 48KB available

x_smem:        2048 * 2 = 4KB    (input activation)
wq_decoded:    32 * 32 * 2 = 2KB (one tile)
wk_decoded:    32 * 32 * 2 = 2KB (one tile)
wv_decoded:    32 * 32 * 2 = 2KB (one tile)
scale_smem:    32 * 2 = 64B      (scale tile)
accumulators:  在 register

Total used: ~10KB, 足够!
```

### Packed FP4 Decode Strategy
```cuda
// Group scale broadcast - 减少 scale lookup
// 每 32 个 FP4 值共享一个 scale

__device__ void decode_tile_fp4(
    const uint8_t* packed,  // [TILE_K * TILE_N / 2]
    const half* scales,     // [TILE_N * (TILE_K / 32)]
    half* decoded           // [TILE_K, TILE_N]
) {
    int tid = threadIdx.x;
    int num_elements = TILE_K * TILE_N / 2;

    for (int i = tid; i < num_elements; i += blockDim.x) {
        int k = (i * 2) / TILE_N;
        int n = (i * 2) % TILE_N;
        int scale_idx = n * (TILE_K / 32) + k / 32;

        half scale = scales[scale_idx];
        uint8_t p = packed[i];

        decoded[k * TILE_N + n] = __hmul(NVFP4_LUT[p & 0xF], scale);
        decoded[(k+1) * TILE_N + n] = __hmul(NVFP4_LUT[p >> 4], scale);
    }
}
```

### KV Store Vectorization
```cuda
// 使用 float4 (8 个 half) 进行 coalesced write

__device__ void store_kv_vectorized(
    half* k_cache,  // [B, max_seq, num_kv_heads, head_dim]
    half* v_cache,
    const half* k_result,  // [head_dim]
    const half* v_result,
    int batch_idx, int cache_pos, int kv_head_idx
) {
    // 每 8 个线程协作写 64 bytes = 32 half values
    int lane = threadIdx.x % 32;

    if (lane < 32) {  // head_dim = 256 = 32 * 8 halfs
        float4* k_ptr = reinterpret_cast<float4*>(
            &k_cache[batch_idx * max_seq * num_kv_heads * head_dim +
                     cache_pos * num_kv_heads * head_dim +
                     kv_head_idx * head_dim +
                     lane * 8]);
        float4* v_ptr = reinterpret_cast<float4*>(
            &v_cache[/* same offset */]);

        *k_ptr = *reinterpret_cast<const float4*>(&k_result[lane * 8]);
        *v_ptr = *reinterpret_cast<const float4*>(&v_result[lane * 8]);
    }
}
```

---

## 验证里程碑

### Week 1
- [ ] Shared memory FP4 decode cache 实现
- [ ] 单独 Q projection FP4 kernel 验证
- [ ] Decode latency 测量 (目标: < 0.05ms)

### Week 2
- [ ] Fused QKV kernel 完成
- [ ] KV cache direct write 实现
- [ ] QKV fuse speedup 验证 (目标: > 1.5x vs baseline)

### Week 3
- [ ] KV persistent kernel prototype
- [ ] Attention + KV fuse 实验
- [ ] End-to-end latency 测量

### Week 4
- [ ] TVM TensorIR integration
- [ ] TRT plugin 封装
- [ ] Production deployment

---

## 预期最终收益

| 优化 | Latency 贡献 | 优化后 |
|------|-------------|--------|
| KV Pipeline (57%) | 57% → 28% | 29% saved |
| MLP (25%) | 保持或微优 | - |
| Others (18%) | 保持 | - |

**整体预期**: 173ms → ~120ms (~1.4x speedup, ~8.3 Hz)

---

## 风险与备选方案

### 风险1: Shared memory decode overhead
**备选**: Fragment-layout packing, 直接MMA-compatible layout

### 风险2: GQA KV broadcast 效率
**备选**: Warp shuffle broadcast

### 风险3: TRT plugin integration 复杂
**备选**: 先用 PyTorch CUDA extension 验证，后续再迁移

---

## 参考实现

- [FlashDecoding](https://github.com/Dao-AILab/flash-attention): KV cache optimization pattern
- [vLLM PagedAttention](https://github.com/vllm-project/vllm): KV cache memory management
- [CUTLASS FP4](https://github.com/NVIDIA/cutlass): Fragment layout reference
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM): Production KV cache integration
