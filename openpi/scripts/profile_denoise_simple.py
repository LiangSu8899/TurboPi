#!/usr/bin/env python3
"""Simple profiling of Denoise - compare with/without CUDA Graph."""

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(workspace_dir, 'src'))
os.environ['OPENPI_SKIP_TVM'] = '1'
os.chdir(workspace_dir)

import torch
import time
import numpy as np

torch.backends.cudnn.enabled = False

print('=' * 70)
print('Denoise 性能对比 (CUDA Graph vs 无 Graph)')
print('=' * 70)

device = torch.device('cuda')

# Load model
from openpi.inference.unified_policy import UnifiedPolicy

print('\n加载模型...')
policy = UnifiedPolicy(
    checkpoint_dir='/root/.cache/openpi/pytorch_checkpoints/pi05_libero',
    backend='pytorch',
    num_denoising_steps=10,
    device='cuda',
)
policy.warmup(num_iterations=2)
model = policy.backend.model

# Prepare test data
test_obs = {
    'observation/image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    'observation/wrist_image': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    'observation/state': np.zeros(8, dtype=np.float32),
    'prompt': 'pick up the bowl',
}

obs = policy.backend._preprocess(test_obs)
images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(obs, train=False)
prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
prefix_kv_cache = model.compute_prefix_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)

action_horizon = model.config.action_horizon
action_dim = model.config.action_dim
prefix_len = prefix_pad_masks.shape[1]

print(f'\n配置:')
print(f'  prefix_len: {prefix_len}')
print(f'  action_horizon: {action_horizon}')
print(f'  action_dim: {action_dim}')
print(f'  num_denoising_steps: 10')

# 1. Test without CUDA Graph (10 steps loop)
print('\n' + '=' * 70)
print('1. 无 CUDA Graph (10 步循环)')
print('=' * 70)

def run_denoise_loop(model, state, x_t, prefix_kv_cache, prefix_pad_masks, num_steps=10):
    """Run denoise loop without CUDA graph."""
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.tensor([1.0 - i * dt], device=x_t.device, dtype=torch.float32)
        with torch.no_grad():
            v_t = model.denoise_step_with_cache(
                state=state,
                prefix_kv_cache=prefix_kv_cache,
                prefix_pad_masks=prefix_pad_masks,
                x_t=x_t,
                timestep=t,
            )
        x_t = x_t + v_t * dt
    return x_t

x_t = torch.randn(1, action_horizon, action_dim, device=device, dtype=torch.bfloat16)

# Warmup
for _ in range(3):
    _ = run_denoise_loop(model, state, x_t.clone(), prefix_kv_cache, prefix_pad_masks)
torch.cuda.synchronize()

# Benchmark
runs = 20
start = time.perf_counter()
for _ in range(runs):
    _ = run_denoise_loop(model, state, x_t.clone(), prefix_kv_cache, prefix_pad_masks)
torch.cuda.synchronize()
no_graph_ms = (time.perf_counter() - start) / runs * 1000

print(f'  无 CUDA Graph: {no_graph_ms:.2f} ms (10 steps)')

# 2. Test with ChainedDenoiseGraphs
print('\n' + '=' * 70)
print('2. 使用 ChainedDenoiseGraphs')
print('=' * 70)

try:
    from openpi.modules.graphed_denoise import ChainedDenoiseGraphs

    # Create and capture
    chained = ChainedDenoiseGraphs(model=model, num_steps=10, device=device)
    chained.capture(state, prefix_kv_cache, prefix_pad_masks, warmup_iters=3)

    # Warmup - forward only takes noise tensor
    for _ in range(5):
        _ = chained.forward(x_t.clone())
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(runs):
        _ = chained.forward(x_t.clone())
    torch.cuda.synchronize()
    chained_ms = (time.perf_counter() - start) / runs * 1000

    print(f'  ChainedDenoiseGraphs: {chained_ms:.2f} ms (10 steps)')
    print(f'  加速: {no_graph_ms / chained_ms:.2f}x')
    print(f'  节省: {no_graph_ms - chained_ms:.2f} ms')

    graph_speedup = no_graph_ms / chained_ms
    graph_savings = no_graph_ms - chained_ms

except Exception as e:
    print(f'  ChainedDenoiseGraphs 失败: {e}')
    import traceback
    traceback.print_exc()
    chained_ms = no_graph_ms
    graph_speedup = 1.0
    graph_savings = 0.0

# 3. CUDA Event profiling for single step
print('\n' + '=' * 70)
print('3. 单步时间分解 (CUDA Events)')
print('=' * 70)

def profile_with_cuda_events(name, func, runs=50):
    """Profile a function using CUDA events."""
    # Warmup
    for _ in range(10):
        func()
    torch.cuda.synchronize()

    # Profile
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]

    for i in range(runs):
        start_events[i].record()
        func()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(runs)]
    return sum(times) / len(times)

# Single step
timestep = torch.ones(1, device=device, dtype=torch.float32)

def single_step():
    with torch.no_grad():
        return model.denoise_step_with_cache(state, prefix_kv_cache, prefix_pad_masks, x_t, timestep)

step_ms = profile_with_cuda_events('single_step', single_step, runs=50)
print(f'  单步 denoise_step_with_cache: {step_ms:.3f} ms')
print(f'  10 步估算: {step_ms * 10:.2f} ms')
print(f'  vs 实际循环: {no_graph_ms:.2f} ms')
print(f'  循环 overhead: {no_graph_ms - step_ms * 10:.2f} ms')

# 4. Summary
print('\n' + '=' * 70)
print('总结')
print('=' * 70)

print(f'''
Denoise 10 步性能对比:
┌────────────────────────────────────────────────────────────────────┐
│ 方案                    │ 耗时        │ 加速     │ 节省        │
├────────────────────────────────────────────────────────────────────┤
│ 无 CUDA Graph (循环)    │ {no_graph_ms:6.2f} ms   │ 1.00x    │ baseline    │
│ ChainedDenoiseGraphs    │ {chained_ms:6.2f} ms   │ {graph_speedup:.2f}x    │ {graph_savings:.2f} ms      │
└────────────────────────────────────────────────────────────────────┘

单步分析:
  - 单步时间: {step_ms:.3f} ms
  - 10 步理论: {step_ms * 10:.2f} ms
  - 实际循环: {no_graph_ms:.2f} ms
  - Python/CUDA overhead: {no_graph_ms - step_ms * 10:.2f} ms ({(no_graph_ms - step_ms * 10) / no_graph_ms * 100:.1f}%)

CUDA Graph 效果:
  - 消除了大部分 kernel launch overhead
  - 10 步链式 graph 比 10 次单独 launch 更快

进一步优化方向:
  1. Graph 内部计算无法通过 Python 层面优化
  2. 需要修改底层 kernel (如 Triton fused kernel)
  3. 或减少计算量 (如减少 prefix_len)
''')
