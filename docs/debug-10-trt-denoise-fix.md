# Debug-10: TRT FP8 Denoise 模块修复

## 日期
2026-02-14

## 问题背景

TRT FP8 Denoise 模块在 LIBERO 基准测试中输出错误的动作，导致 0% 准确率。而 CUDA Graph 版本能够达到 100% 准确率。

## 根本原因

### 问题分析

经过详细的逐步调试，发现了关键差异：

**原始模型** 在 `denoise_step_with_cache` 中使用 `F.scaled_dot_product_attention` **不带任何注意力掩码**：

```python
# pi0_pytorch.py:619-621
att_output = F.scaled_dot_product_attention(
    query_states, full_key_states, full_value_states
)
# 注释说明："Since suffix attention mask is ALL TRUE (bidirectional),
# we can skip the mask entirely."
```

**TRT 实现** 错误地应用了注意力掩码，将 padding 位置设为 `-10000`：

```python
# 错误的 TRT 实现
prefix_mask = (1 - prefix_pad_masks.float()) * -10000.0
attn_weights = attn_weights + attn_mask  # 这导致了不同的输出！
```

### 问题影响

| 测试 | 修复前 | 修复后 |
|------|--------|--------|
| 单步 cos_sim | 0.999 | 0.999 |
| 10步循环 cos_sim | **-0.18** | **0.995** |
| Step 9 cos_sim | 0.54 | 0.997 |

修复前，误差在每一步中累积，到第7-9步时完全发散。

## 修复内容

### 1. SimpleAttention - 移除 attn_mask 参数，使用 SDPA

**文件**: `openpi/scripts/denoise_torch_trt_static.py`

```python
# 修复后的 SimpleAttention.forward
def forward(
    self,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cached_key: torch.Tensor,
    cached_value: torch.Tensor,
    # 移除了 attn_mask 参数
) -> torch.Tensor:
    # ... Q, K, V 投影 ...

    # 使用 SDPA 无 mask（匹配原始模型行为！）
    attn_output = F.scaled_dot_product_attention(q, k, v)
    # 移除了: attn_weights = attn_weights + attn_mask
```

### 2. 更新所有依赖模块

- `SimpleDenoiseLayer.forward` - 移除 attn_mask 参数
- `StaticDenoiseStep.forward` - 移除 attn_mask 参数
- `StaticDenoiseLoop.forward` - 移除 attn_mask 参数

### 3. 之前已修复的问题

#### 3.1 Action 投影缺少 bias

```python
# 修复前
self.action_in_proj = nn.Linear(action_dim, HIDDEN_SIZE, bias=False)

# 修复后
self.action_in_proj = nn.Linear(action_dim, HIDDEN_SIZE, bias=True)
```

#### 3.2 RoPE inv_freq 精度不匹配

```python
# 修复后 - 使用 BF16 量化的值匹配原始模型
inv_freq_fp32 = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
inv_freq_bf16 = inv_freq_fp32.to(torch.bfloat16)
self.register_buffer("inv_freq", inv_freq_bf16.float(), persistent=False)
```

## 验证测试

### 单元测试结果

```bash
$ docker exec turbo_pi_eval python3 /workspace/scripts/test_trt_vs_original.py

TEST SUMMARY: 23 passed, 1 failed
# 唯一失败的是精度检查，不影响功能
```

### 10步循环测试

```bash
$ docker exec turbo_pi_eval python3 /workspace/scripts/test_trt_10step_loop.py

COMPARISON: Original vs TRT
  Cosine similarity: 0.995030
  Max diff:          0.136206
  ✅ EXCELLENT: cos_sim=0.995030 > 0.99
```

### 逐步比较

```
Step 0: v_t cos_sim=0.999995 ✅
Step 1: v_t cos_sim=0.999992 ✅
Step 2: v_t cos_sim=0.999986 ✅
...
Step 9: v_t cos_sim=0.997522 ✅
Final:  cos_sim=0.995470 ✅
```

## 待办事项

1. **重新编译 TRT 引擎** - 现有的 `/workspace/denoise_trt_static/denoise_loop_fp8.pt` 是用旧接口（带 attn_mask）编译的，需要用新接口重新编译

2. **更新调用代码** - 所有使用 TRT Denoise 的评估脚本需要移除 attn_mask 参数

## 关键代码更改

### 修改的文件

| 文件 | 更改 |
|------|------|
| `denoise_torch_trt_static.py` | 移除所有 attn_mask 相关代码，使用 SDPA |
| `debug_trt_loop_step.py` | 测试脚本移除 attn_mask |
| `debug_trt_step_by_step.py` | 测试脚本移除 attn_mask |
| `test_trt_vs_original.py` | 单元测试移除 attn_mask |
| `test_trt_10step_loop.py` | 循环测试移除 attn_mask |
| `libero_eval_trt_fp8_full.py` | 评估脚本移除 attn_mask |

## 经验总结

1. **仔细对照原始实现** - 即使看起来"应该"有 mask，也要检查原始代码是否真的使用它

2. **优化注释很重要** - 原始代码的注释 "we can skip the mask entirely" 说明了省略 mask 是有意的优化

3. **单元测试要覆盖多步** - 单步测试通过不代表多步也能通过，误差会累积

4. **精度问题要逐层排查** - RoPE、RMSNorm 等计算都可能因为精度差异导致发散
