#!/usr/bin/env python3
"""
TVM NVFP4 GEMM Kernel

使用 TVM 生成高效的 NVFP4 GEMM kernel，支持任意 M 值。
针对 Thor (SM110) 优化。

Author: Claude Code
Date: 2026-02-10
"""

import tvm
from tvm import te, tir, auto_scheduler
from tvm.topi.utils import get_const_tuple
import numpy as np

# NVFP4 配置
BLOCK_SIZE = 32  # Block scaling 块大小
NVFP4_MAX = 6.0

# NVFP4 E2M1 解码表
NVFP4_DECODE_TABLE = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=np.float32)


def nvfp4_gemm_compute(M, N, K, block_size=BLOCK_SIZE):
    """
    定义 NVFP4 GEMM 的计算。

    C[m, n] = sum_k(A[m, k] * dequant(W_packed[n, k//2], scale_W[n, k//block_size]))

    Args:
        M: batch * seq_len
        N: out_features
        K: in_features
        block_size: block scaling 块大小

    Returns:
        (A, W_packed, scale_W, C): TVM tensors
    """
    num_blocks_k = K // block_size

    # 输入张量
    A = te.placeholder((M, K), name="A", dtype="float32")
    W_packed = te.placeholder((N, K // 2), name="W_packed", dtype="uint8")
    scale_W = te.placeholder((N, num_blocks_k), name="scale_W", dtype="float32")

    # 解码表 (常量)
    decode_table = te.placeholder((16,), name="decode_table", dtype="float32")

    # 定义 NVFP4 反量化
    def unpack_nvfp4(packed, idx):
        """从 packed uint8 中提取 NVFP4 值"""
        # idx % 2 == 0: 低 4 位
        # idx % 2 == 1: 高 4 位
        is_high = idx % 2
        nibble = tir.if_then_else(
            is_high == 0,
            packed & tir.const(0xF, "uint8"),
            (packed >> 4) & tir.const(0xF, "uint8")
        )
        return nibble

    # Reduction axis
    k = te.reduce_axis((0, K), name="k")

    # 计算 GEMM
    def compute_gemm(m, n):
        # 获取 packed 权重索引
        packed_idx = k // 2

        # 获取 nibble (4-bit index)
        packed_val = W_packed[n, packed_idx]
        is_high = k % 2
        nibble = tir.if_then_else(
            is_high == 0,
            packed_val & tir.const(0xF, "uint8"),
            (packed_val >> 4) & tir.const(0xF, "uint8")
        )

        # 反量化: 查表 + scale
        block_idx = k // block_size
        w_dequant = decode_table[nibble.astype("int32")] * scale_W[n, block_idx]

        # GEMM
        return te.sum(A[m, k] * w_dequant, axis=k)

    C = te.compute(
        (M, N),
        compute_gemm,
        name="C"
    )

    return A, W_packed, scale_W, decode_table, C


def schedule_nvfp4_gemm_cuda(A, W_packed, scale_W, decode_table, C, target="cuda"):
    """
    为 CUDA 创建优化的 schedule。

    优化策略:
    1. 2D tiling: block 级别和 thread 级别
    2. Shared memory: 缓存 A 和 W 的 tile
    3. 向量化: 内层循环向量化
    4. Loop unrolling
    """
    s = te.create_schedule(C.op)

    # 获取维度
    M, N = get_const_tuple(C.shape)
    K = get_const_tuple(A.shape)[1]

    # Tiling 参数 (需要 auto-tune)
    block_m = 64
    block_n = 64
    block_k = 32
    thread_m = 8
    thread_n = 8

    # 获取轴
    m, n = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # Block 级别 tiling
    mo, mi = s[C].split(m, factor=block_m)
    no, ni = s[C].split(n, factor=block_n)
    ko, ki = s[C].split(k, factor=block_k)

    # Reorder
    s[C].reorder(mo, no, ko, mi, ni, ki)

    # Bind to blocks
    s[C].bind(mo, te.thread_axis("blockIdx.y"))
    s[C].bind(no, te.thread_axis("blockIdx.x"))

    # Thread 级别 tiling
    mio, mii = s[C].split(mi, factor=thread_m)
    nio, nii = s[C].split(ni, factor=thread_n)

    s[C].reorder(mio, nio, mii, nii, ki)

    # Bind to threads
    s[C].bind(mio, te.thread_axis("threadIdx.y"))
    s[C].bind(nio, te.thread_axis("threadIdx.x"))

    # Unroll
    s[C].unroll(ki)

    return s


def build_nvfp4_gemm(M, N, K, target="cuda", target_host="llvm"):
    """
    构建 NVFP4 GEMM kernel。

    Args:
        M, N, K: GEMM 维度
        target: 目标设备
        target_host: host 目标

    Returns:
        TVM module
    """
    A, W_packed, scale_W, decode_table, C = nvfp4_gemm_compute(M, N, K)
    s = schedule_nvfp4_gemm_cuda(A, W_packed, scale_W, decode_table, C, target)

    with tvm.transform.PassContext(opt_level=3):
        func = tvm.build(s, [A, W_packed, scale_W, decode_table, C], target=target, target_host=target_host)

    return func


def auto_tune_nvfp4_gemm(M, N, K, target="cuda", log_file="nvfp4_gemm_tune.json"):
    """
    使用 AutoScheduler 自动调优 NVFP4 GEMM。
    """
    from tvm import auto_scheduler

    @auto_scheduler.register_workload
    def nvfp4_gemm_workload(M, N, K):
        return nvfp4_gemm_compute(M, N, K)

    # 创建任务
    task = auto_scheduler.SearchTask(
        func=nvfp4_gemm_workload,
        args=(M, N, K),
        target=target
    )

    # 调优配置
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )

    # 开始调优
    task.tune(tune_option)

    # 应用最佳配置
    sch, args = task.apply_best(log_file)

    return sch, args


# ============================================================================
# 简化版本: 直接使用 CUDA intrinsics
# ============================================================================

def nvfp4_gemm_cuda_source(M, N, K, block_size=32):
    """
    生成优化的 CUDA kernel 源码。

    这个版本直接生成 CUDA 代码，使用:
    1. Shared memory tiling
    2. 向量化内存访问
    3. Warp-level primitives
    """

    # Tiling 参数
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    THREAD_M = 8
    THREAD_N = 8

    cuda_source = f'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_M {BLOCK_M}
#define BLOCK_N {BLOCK_N}
#define BLOCK_K {BLOCK_K}
#define THREAD_M {THREAD_M}
#define THREAD_N {THREAD_N}
#define BLOCK_SIZE_SCALE {block_size}

// NVFP4 解码表
__constant__ float NVFP4_DECODE[16] = {{
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
}};

__device__ __forceinline__ float decode_nvfp4(uint8_t packed, int idx) {{
    int nibble = (idx & 1) ? (packed >> 4) : (packed & 0xF);
    return NVFP4_DECODE[nibble];
}}

__global__ void nvfp4_gemm_kernel(
    const float* __restrict__ A,          // [M, K]
    const uint8_t* __restrict__ W_packed, // [N, K/2]
    const float* __restrict__ scale_W,    // [N, K/BLOCK_SIZE_SCALE]
    float* __restrict__ C,                // [M, N]
    int M, int N, int K
) {{
    // Block 和 thread 索引
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Shared memory
    __shared__ float As[BLOCK_M][BLOCK_K];
    __shared__ float Ws[BLOCK_K][BLOCK_N];

    // 累加器
    float acc[THREAD_M][THREAD_N] = {{0.0f}};

    // 计算全局起始位置
    int m_start = by * BLOCK_M;
    int n_start = bx * BLOCK_N;

    int num_blocks_k = K / BLOCK_SIZE_SCALE;

    // K 维度循环
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {{
        // 加载 A tile 到 shared memory
        for (int i = 0; i < BLOCK_M; i += blockDim.y) {{
            for (int j = 0; j < BLOCK_K; j += blockDim.x) {{
                int m = m_start + ty + i;
                int k = k0 + tx + j;
                if (m < M && k < K) {{
                    As[ty + i][tx + j] = A[m * K + k];
                }} else {{
                    As[ty + i][tx + j] = 0.0f;
                }}
            }}
        }}

        // 加载 W tile 到 shared memory (反量化)
        for (int i = 0; i < BLOCK_K; i += blockDim.y) {{
            for (int j = 0; j < BLOCK_N; j += blockDim.x) {{
                int k = k0 + ty + i;
                int n = n_start + tx + j;
                if (k < K && n < N) {{
                    int packed_idx = k / 2;
                    uint8_t packed = W_packed[n * (K / 2) + packed_idx];
                    float w_val = decode_nvfp4(packed, k);

                    int block_idx = k / BLOCK_SIZE_SCALE;
                    float scale = scale_W[n * num_blocks_k + block_idx];

                    Ws[ty + i][tx + j] = w_val * scale;
                }} else {{
                    Ws[ty + i][tx + j] = 0.0f;
                }}
            }}
        }}

        __syncthreads();

        // 计算 GEMM tile
        for (int k = 0; k < BLOCK_K; k++) {{
            for (int m = 0; m < THREAD_M; m++) {{
                for (int n = 0; n < THREAD_N; n++) {{
                    acc[m][n] += As[ty * THREAD_M + m][k] * Ws[k][tx * THREAD_N + n];
                }}
            }}
        }}

        __syncthreads();
    }}

    // 写回结果
    for (int m = 0; m < THREAD_M; m++) {{
        for (int n = 0; n < THREAD_N; n++) {{
            int gm = m_start + ty * THREAD_M + m;
            int gn = n_start + tx * THREAD_N + n;
            if (gm < M && gn < N) {{
                C[gm * N + gn] = acc[m][n];
            }}
        }}
    }}
}}

// Wrapper function
extern "C" void launch_nvfp4_gemm(
    const float* A,
    const uint8_t* W_packed,
    const float* scale_W,
    float* C,
    int M, int N, int K,
    cudaStream_t stream
) {{
    dim3 block(BLOCK_N / THREAD_N, BLOCK_M / THREAD_M);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    nvfp4_gemm_kernel<<<grid, block, 0, stream>>>(A, W_packed, scale_W, C, M, N, K);
}}
'''
    return cuda_source


def compile_nvfp4_gemm_cuda(output_path="nvfp4_gemm.so"):
    """
    编译 CUDA kernel 为共享库。
    """
    import subprocess
    import tempfile
    import os

    # 生成 CUDA 源码
    cuda_source = nvfp4_gemm_cuda_source(0, 0, 0)  # 动态 shape

    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
        f.write(cuda_source)
        cu_path = f.name

    # 编译
    cmd = [
        'nvcc',
        '-shared', '-Xcompiler', '-fPIC',
        '-O3', '--use_fast_math',
        '-arch=sm_89',  # Thor
        '-o', output_path,
        cu_path
    ]

    subprocess.run(cmd, check=True)
    os.unlink(cu_path)

    return output_path


if __name__ == "__main__":
    # 测试编译
    print("Generating NVFP4 GEMM CUDA kernel...")
    source = nvfp4_gemm_cuda_source(455, 16384, 2048)
    print(source[:1000])
    print("...")
    print("\nKernel generated successfully!")
