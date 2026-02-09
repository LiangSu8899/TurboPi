---

## 4. 实验验证结果 (2026-02-09)

### 4.1 量化方案精度对比

在 Thor SM110 上进行了完整的 Pi0.5 模型精度测试：

| Method | 权重 | 激活 | Cosine Sim | MAE | 状态 |
|--------|------|------|-----------|-----|------|
| BF16 (baseline) | BF16 | BF16 | 1.0000 | 0.0000 | 基准 |
| **W4A16** | NVFP4 | BF16 | **0.9997** | 0.0022 | ✅ 推荐 |
| **W4A8** | NVFP4 | FP8 | **0.9997** | 0.0022 | ✅ 推荐 |
| W4A4 (Sim) | NVFP4 | NVFP4 | 0.9998 | - | ⚠️ 太慢 |
| W4A4 (CUTLASS) | NVFP4 | NVFP4 | 0.9998 | - | ⚠️ 太慢 |

**关键发现**：
- W4A16 和 W4A8 精度完全满足 Diffusion Policy 的 0.98 要求
- W4A4 精度也可接受，但在线量化开销太大

### 4.2 速度对比

| Method | Time (ms) | Hz | Speedup | 瓶颈 |
|--------|-----------|-----|---------|------|
| BF16 | 175.3 | 5.70 | 1.0x | - |
| W4A16 (cached) | 174.3 | 5.74 | 1.0x | 无加速（已反量化缓存） |
| W4A8 (cached) | 172.7 | 5.79 | 1.0x | 无加速（已反量化缓存） |
| **W4A4 CUTLASS** | **7401.5** | **0.14** | **0.02x** | 在线量化 7.6ms/层 |

**问题根源**：
```
W4A4 当前流程:
Input (BF16) -> [quantize_to_nvfp4_sim: 7.6ms] -> CUTLASS GEMM: 0.24ms
                  ↑ Python 在线量化太慢！
```

### 4.3 FP8 Scale Overflow 问题

测试发现 Layer 16/17 的 `down_proj` 激活值超出 FP8 Scale 范围：

| Layer | Min | Max | |Max| | 阈值 (2688) | 状态 |
|-------|-----|-----|-------|--------------|------|
| layer16.down_proj | -6784 | 548 | 6784 | 2688 | ⚠️ 超过 |
| layer17.down_proj | -3648 | 3904 | 3904 | 2688 | ⚠️ 超过 |

**原因分析**：
- FP8 E4M3 最大值 = 448
- NVFP4 最大值 = 6
- Scale 最大值 = 448 × 6 = 2688
- 当 |activation| > 2688 时，scale 会溢出导致 NaN

**解决方案**：添加 Clamp 限制输入范围
```python
FP8_SCALE_MAX = 448.0 * 6.0  # 2688
x = x.clamp(-FP8_SCALE_MAX, FP8_SCALE_MAX)
```

Clamp 后精度仍保持 0.9998，说明超阈值的值是少数 outliers。

### 4.4 方案对比总结

| 方案 | 精度 | 理论速度 | 开发难度 | 推荐度 |
|------|------|----------|----------|--------|
| W4A4 (当前实现) | 0.9998 | 0.14 Hz | 已完成 | ❌ 不可用 |
| W4A16 (cached) | 0.9997 | 5.7 Hz | 已完成 | ⚠️ 无加速 |
| **W4A16 (kernel)** | 0.9997 | **10+ Hz** | 需写 CUDA | ⭐⭐⭐ 最佳 |
| W4A8 (kernel) | 0.9997 | 8+ Hz | 需写 CUDA | ⭐⭐ 次选 |

---

## 5. 推荐实现路径：W4A16 CUDA Kernel

### 5.1 为什么选择 W4A16

1. **精度最佳** (0.9997) - 无激活量化误差
2. **零激活量化开销** - 不需要 7.6ms 的在线量化
3. **75% 权重带宽节省** - 从显存加载 4-bit 权重
4. **开发简单** - 只需实现权重 dequantize，使用标准 BF16 Tensor Core

### 5.2 Kernel 设计

```
目标流程:
Weight (NVFP4, 4-bit) ─────────────────────────────────────┐
                                                           ↓
                          [Load 4-bit] → [Dequant to BF16 in registers]
                                                           ↓
Activation (BF16) ──────────────────────────────→ [BF16 Tensor Core GEMM]
                                                           ↓
                                                     Output (BF16)
```

### 5.3 Triton 实现思路

```python
@triton.jit
def w4a16_gemm_kernel(
    # Inputs
    A_ptr,              # Activation (BF16) [M, K]
    B_packed_ptr,       # Weight packed (NVFP4) [N, K//2]
    B_scales_ptr,       # Weight scales (FP8/FP32) [N, K//BLOCK]
    C_ptr,              # Output (BF16) [M, N]
    # Dimensions
    M, N, K,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 1. 计算 Block 位置
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. 加载 Activation Block (BF16)
    # A: [BLOCK_M, BLOCK_K]
    a = tl.load(A_ptr + ...)

    # 3. 加载 Weight Block (NVFP4 packed)
    # B_packed: [BLOCK_N, BLOCK_K//2]
    b_packed = tl.load(B_packed_ptr + ...)

    # 4. 在寄存器中解压 NVFP4 -> BF16
    # 这是关键步骤：被搬运延迟掩盖
    b_unpacked = dequant_nvfp4_to_bf16(b_packed, B_scales_ptr)

    # 5. BF16 矩阵乘法 (Tensor Core)
    c = tl.dot(a, b_unpacked)

    # 6. 存储结果
    tl.store(C_ptr + ..., c)
```

### 5.4 预期收益

| 指标 | BF16 Baseline | W4A16 Kernel (预期) |
|------|--------------|---------------------|
| 权重带宽 | 100% | **25%** (-75%) |
| 激活带宽 | 100% | 100% (无变化) |
| 总带宽 | 100% | ~50% |
| 推理速度 | 5.7 Hz | **10-12 Hz** |
| 精度 | 1.0000 | 0.9997 |

---

## 6. 实验代码位置

测试脚本（位于 `openpi/scripts/`）：

| 脚本 | 功能 |
|------|------|
| `compare_quantization_methods.py` | W4A4/W4A8/W4A16 精度对比 |
| `benchmark_w4a4_cutlass.py` | W4A4 CUTLASS 速度测试 |
| `test_w4a4_modes.py` | Simulation vs CUTLASS 模式对比 |
| `check_activation_range.py` | 激活值范围检测 (FP8 Overflow) |

量化模块（位于 `openpi/src/openpi/models_pytorch/`）：

| 模块 | 功能 |
|------|------|
| `nvfp4_mlp.py` | NVFP4 W4A4 实现 (含 CUTLASS) |
| `w4a16_mlp.py` | W4A16 实现 (cached dequant) |
| `w4a8_mlp.py` | W4A8 实现 |

---

## 7. 下一步行动

1. **实现 W4A16 Triton/CUDA Kernel**
   - Load 4-bit weight from memory
   - Dequant to BF16 in registers (fused with load)
   - BF16 Tensor Core GEMM

2. **目标性能**
   - 精度: 0.99+
   - 速度: 10+ Hz (2x 加速)

3. **验证路径**
   - 单层精度验证
   - 全模型精度验证
   - LIBERO 任务成功率验证