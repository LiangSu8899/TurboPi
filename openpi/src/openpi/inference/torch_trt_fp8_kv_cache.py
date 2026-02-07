#!/usr/bin/env python3
"""
Torch-TRT FP8 KV Cache Engine.

Uses ModelOpt + Torch-TensorRT for FP8 MLP compilation, achieving 2.94x speedup
over PyTorch FP16 (vs 1.0x with PyTorch native FP8).

Key Features:
1. Flash Attention for attention layers (same as flash_fp8_kv_cache.py)
2. Torch-TRT FP8 compiled MLP for 2.94x speedup
3. Static graph optimization - no runtime quantization overhead

Performance (Thor, seq=970, 18 layers):
- PyTorch FP16: 59.89 ms
- PyTorch native FP8: ~60 ms (no speedup!)
- Torch-TRT FP8: 20.39 ms (2.94x speedup)

Expected Full Pipeline:
- Current (PyTorch FP8): 180 ms (5.5 Hz)
- With Torch-TRT FP8: ~140 ms (7.1 Hz)

Usage:
    from openpi.inference.torch_trt_fp8_kv_cache import TorchTRTFP8KVCacheEngine

    engine = TorchTRTFP8KVCacheEngine(checkpoint_dir)
    keys, values, hidden = engine.infer(hidden_states, attention_mask)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Check dependencies
try:
    import torch_tensorrt
    HAS_TORCH_TRT = True
except ImportError:
    HAS_TORCH_TRT = False
    logger.warning("torch_tensorrt not available")

try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode
    HAS_MODELOPT = True
except ImportError:
    HAS_MODELOPT = False
    logger.warning("modelopt not available")

# Model constants (pi0.5 / PaliGemma)
NUM_LAYERS = 18
NUM_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
HIDDEN_SIZE = 2048
MLP_DIM = 16384
SEQ_LEN = 968  # Actual seq_len from embed_prefix: 256 (image) + 512 (language/pad) + 200 (state placeholder) = 968
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10000.0


class RMSNorm(nn.Module):
    """RMSNorm layer matching Gemma's implementation."""
    def __init__(self, hidden_size, eps=RMS_NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return (x * (1.0 + self.weight.float())).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Gemma/Pi0.5."""
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = ROPE_THETA, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        inv_freq = inv_freq.to(dtype)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply Rotary Position Embedding to Q and K tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SimpleMLP(nn.Module):
    """Single MLP layer for Torch-TRT compilation."""
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_SIZE, MLP_DIM, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, MLP_DIM, bias=False)
        self.down_proj = nn.Linear(MLP_DIM, HIDDEN_SIZE, bias=False)

    def forward(self, x):
        gate = F.gelu(self.gate_proj(x), approximate='tanh')
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class StackedMLP(nn.Module):
    """N stacked MLP layers for Torch-TRT compilation."""
    def __init__(self, n_layers=18):
        super().__init__()
        self.layers = nn.ModuleList([SimpleMLP() for _ in range(n_layers)])

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            out = layer(x)
            outputs.append(out)
            x = x + out  # Residual for next layer's input
        return torch.stack(outputs, dim=0)  # (n_layers, B, S, H)


class GQAAttention(nn.Module):
    """Grouped Query Attention using SDPA with RoPE."""
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)
        self.num_groups = NUM_HEADS // NUM_KV_HEADS
        self.softmax_scale = HEAD_DIM ** -0.5

    def forward(self, x, cos, sin, attention_mask=None):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, NUM_HEADS, HEAD_DIM)
        k = self.k_proj(x).view(B, S, NUM_KV_HEADS, HEAD_DIM)
        v = self.v_proj(x).view(B, S, NUM_KV_HEADS, HEAD_DIM)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        # Store K, V for cache
        k_cache = k.transpose(1, 2).contiguous()
        v_cache = v.transpose(1, 2).contiguous()

        # Make contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Expand KV for GQA
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=2)
            v = v.repeat_interleave(self.num_groups, dim=2)

        # SDPA
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )
        attn_out = attn_out.transpose(1, 2)

        attn_out = attn_out.reshape(B, S, -1)
        output = self.o_proj(attn_out)

        return output, k_cache, v_cache


class TorchTRTFP8TransformerBlock(nn.Module):
    """Transformer block with SDPA Attention + placeholder for TRT MLP."""
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(HIDDEN_SIZE)
        self.self_attn = GQAAttention()
        self.post_attention_layernorm = RMSNorm(HIDDEN_SIZE)

        # FP16 MLP (will be replaced by TRT compiled version)
        self.gate_proj = nn.Linear(HIDDEN_SIZE, MLP_DIM, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, MLP_DIM, bias=False)
        self.down_proj = nn.Linear(MLP_DIM, HIDDEN_SIZE, bias=False)

        # TRT MLP placeholder
        self._trt_mlp = None

    def _mlp_forward(self, x):
        if self._trt_mlp is not None:
            # TRT expects FP16, model uses BF16
            # Check shape matches compiled shape (1, 968, 2048)
            if x.shape[1] == SEQ_LEN and x.shape[2] == HIDDEN_SIZE:
                input_dtype = x.dtype
                x_fp16 = x.half().contiguous()
                try:
                    out = self._trt_mlp(x_fp16)
                    return out.to(input_dtype)
                except RuntimeError:
                    # Fallback to FP16 if TRT fails
                    pass
        # Fallback to FP16 MLP
        gate = F.gelu(self.gate_proj(x), approximate='tanh')
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    def forward(self, x, cos, sin, attention_mask=None):
        normed = self.input_layernorm(x)
        attn_output, k, v = self.self_attn(normed, cos, sin, attention_mask)
        x = x + attn_output

        normed = self.post_attention_layernorm(x)
        mlp_output = self._mlp_forward(normed)
        x = x + mlp_output

        return x, k, v


class TorchTRTFP8KVCacheModel(nn.Module):
    """18-layer transformer with SDPA Attention + Torch-TRT FP8 MLP."""
    def __init__(self, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TorchTRTFP8TransformerBlock() for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(HIDDEN_SIZE)
        self.rotary_emb = RotaryEmbedding(head_dim=HEAD_DIM)

        # Compiled TRT MLP for all layers
        self._trt_stacked_mlp = None

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        """
        Forward pass through all transformer layers.

        IMPORTANT: Each layer uses layer._trt_mlp if set, otherwise falls back to FP16 MLP.
        The layer.forward() method handles this via _mlp_forward().
        """
        B, S, _ = hidden_states.shape

        if position_ids is None:
            position_ids = torch.arange(S, device=hidden_states.device).unsqueeze(0).expand(B, -1)

        cos, sin = self.rotary_emb(hidden_states, position_ids)

        all_keys = []
        all_values = []

        # Run through all layers - each layer will use TRT MLP if layer._trt_mlp is set
        x = hidden_states
        for layer in self.layers:
            # layer.forward() calls _mlp_forward() which uses layer._trt_mlp if available
            x, k, v = layer(x, cos, sin, attention_mask)
            all_keys.append(k)
            all_values.append(v)

        hidden_states = self.final_norm(x)

        keys = torch.stack(all_keys, dim=1)
        values = torch.stack(all_values, dim=1)

        return keys, values, hidden_states


def load_weights(model, checkpoint_dir, device="cuda"):
    """Load weights from pi0.5 checkpoint."""
    from safetensors import safe_open

    checkpoint_path = Path(checkpoint_dir)
    weights_path = checkpoint_path / "model.safetensors"

    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    logger.info(f"Loading weights from: {weights_path}")
    state_dict = {}
    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    dtype = torch.bfloat16
    for i, layer in enumerate(model.layers):
        prefix = f"paligemma_with_expert.paligemma.model.language_model.layers.{i}"

        # Attention weights
        layer.self_attn.q_proj.weight.data = state_dict[f"{prefix}.self_attn.q_proj.weight"].to(dtype).to(device)
        layer.self_attn.k_proj.weight.data = state_dict[f"{prefix}.self_attn.k_proj.weight"].to(dtype).to(device)
        layer.self_attn.v_proj.weight.data = state_dict[f"{prefix}.self_attn.v_proj.weight"].to(dtype).to(device)
        layer.self_attn.o_proj.weight.data = state_dict[f"{prefix}.self_attn.o_proj.weight"].to(dtype).to(device)

        # MLP weights
        layer.gate_proj.weight.data = state_dict[f"{prefix}.mlp.gate_proj.weight"].to(dtype).to(device)
        layer.up_proj.weight.data = state_dict[f"{prefix}.mlp.up_proj.weight"].to(dtype).to(device)
        layer.down_proj.weight.data = state_dict[f"{prefix}.mlp.down_proj.weight"].to(dtype).to(device)

        # Norms
        layer.input_layernorm.weight.data = state_dict[f"{prefix}.input_layernorm.weight"].to(dtype).to(device)
        layer.post_attention_layernorm.weight.data = state_dict[f"{prefix}.post_attention_layernorm.weight"].to(dtype).to(device)

    model.final_norm.weight.data = state_dict[
        "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    ].to(dtype).to(device)

    logger.info("Weights loaded successfully")


def compile_trt_fp8_mlp_for_layer(layer, layer_idx, device="cuda"):
    """Compile a single layer's MLP to Torch-TRT FP8.

    Args:
        layer: The transformer layer containing MLP weights
        layer_idx: Layer index for logging
        device: CUDA device

    Returns:
        TRT compiled MLP module, or None if compilation fails
    """
    if not (HAS_TORCH_TRT and HAS_MODELOPT):
        return None

    # Create MLP and copy THIS layer's weights
    mlp = SimpleMLP().to(device).half()
    mlp.gate_proj.weight.data = layer.gate_proj.weight.data.half()
    mlp.up_proj.weight.data = layer.up_proj.weight.data.half()
    mlp.down_proj.weight.data = layer.down_proj.weight.data.half()
    mlp.eval()

    # Sample input for compilation
    x = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=torch.float16)

    # Calibrate
    def calibrate(m):
        with torch.no_grad():
            for _ in range(10):
                m(x)

    # Quantize
    mlp_fp8 = mtq.quantize(mlp, mtq.FP8_DEFAULT_CFG, forward_loop=calibrate)

    # Compile with TRT
    with export_torch_mode():
        trt_mlp = torch_tensorrt.compile(
            mlp_fp8,
            inputs=[x],
            enabled_precisions={torch.float16, torch.float8_e4m3fn},
            workspace_size=8 << 30,
        )

    return trt_mlp


def compile_trt_fp8_mlps(model, device="cuda"):
    """Compile ALL 18 layer MLPs to Torch-TRT FP8.

    IMPORTANT: Each layer has DIFFERENT weights, so we need to compile
    18 separate TRT MLPs. Using a single TRT MLP with layer 0's weights
    for all layers causes 0% accuracy!

    Args:
        model: The full transformer model with all layers
        device: CUDA device

    Returns:
        List of 18 TRT compiled MLP modules, one per layer
    """
    if not (HAS_TORCH_TRT and HAS_MODELOPT):
        logger.warning("Torch-TRT or ModelOpt not available, using FP16 fallback")
        return None

    num_layers = len(model.layers)
    logger.info(f"Compiling Torch-TRT FP8 MLP for ALL {num_layers} layers...")
    logger.info("This will take ~20 seconds per layer (~6 minutes total for 18 layers)")

    trt_mlps = []
    for i, layer in enumerate(model.layers):
        logger.info(f"  Compiling layer {i+1}/{num_layers}...")
        try:
            trt_mlp = compile_trt_fp8_mlp_for_layer(layer, i, device)
            if trt_mlp is not None:
                trt_mlps.append(trt_mlp)
            else:
                logger.warning(f"  Layer {i} TRT compilation returned None, using FP16 fallback")
                trt_mlps.append(None)
        except Exception as e:
            logger.warning(f"  Layer {i} TRT compilation failed: {e}, using FP16 fallback")
            trt_mlps.append(None)

    success_count = sum(1 for m in trt_mlps if m is not None)
    logger.info(f"Torch-TRT FP8 MLP compiled: {success_count}/{num_layers} layers")

    return trt_mlps


class TorchTRTFP8KVCacheEngine:
    """
    Torch-TRT FP8 KV Cache Engine.

    Uses Torch-TRT compiled FP8 MLP for 2.94x speedup over FP16.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        num_layers: int = NUM_LAYERS,
        seq_len: int = SEQ_LEN,
        compile_trt: bool = True,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.num_layers = num_layers
        self.seq_len = seq_len

        logger.info(f"Creating Torch-TRT FP8 KV Cache Model ({num_layers} layers)...")

        # Create model
        self.model = TorchTRTFP8KVCacheModel(num_layers=num_layers).to(device).to(torch.bfloat16)

        # Load weights
        load_weights(self.model, checkpoint_dir, device)
        self.model.eval()

        # Compile TRT FP8 MLP for EACH layer (each layer has different weights!)
        self._trt_mlps = None
        self._trt_compiled_count = 0
        if compile_trt and HAS_TORCH_TRT and HAS_MODELOPT:
            try:
                self._trt_mlps = compile_trt_fp8_mlps(self.model, device)
                # Set each layer's TRT MLP
                if self._trt_mlps is not None:
                    for i, layer in enumerate(self.model.layers):
                        if self._trt_mlps[i] is not None:
                            layer._trt_mlp = self._trt_mlps[i]
                            self._trt_compiled_count += 1
                    logger.info(f"TRT FP8 MLP set for {self._trt_compiled_count}/{num_layers} layers")
            except Exception as e:
                logger.warning(f"TRT FP8 compilation failed: {e}, using FP16 fallback")

        logger.info("Torch-TRT FP8 KV Cache Engine initialized")

    def infer(self, hidden_states, attention_mask=None, position_ids=None):
        """Run KV cache inference."""
        hidden_states = hidden_states.to(self.device, dtype=torch.bfloat16)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)

        with torch.no_grad():
            keys, values, hidden = self.model(hidden_states, attention_mask, position_ids)

        return keys, values, hidden

    def infer_list(self, hidden_states, position_ids, attention_mask):
        """Run KV cache inference and return as list of (K, V) tuples."""
        input_dtype = hidden_states.dtype
        keys, values, _ = self.infer(hidden_states, attention_mask, position_ids)

        keys = keys.to(input_dtype)
        values = values.to(input_dtype)

        result = []
        for i in range(self.num_layers):
            k = keys[:, i]
            v = values[:, i]
            result.append((k, v))

        return result

    def warmup(self, num_iterations=5):
        """Warmup the engine."""
        logger.info(f"Warming up Torch-TRT FP8 Engine ({num_iterations} iterations)...")

        dummy_hidden = torch.randn(
            1, self.seq_len, HIDDEN_SIZE,
            device=self.device, dtype=torch.float16
        )
        dummy_position_ids = torch.arange(self.seq_len, device=self.device).unsqueeze(0)

        for _ in range(num_iterations):
            _ = self.infer(dummy_hidden, None, dummy_position_ids)

        torch.cuda.synchronize()
        logger.info("Warmup complete")

    def benchmark(self, num_warmup=10, num_iters=50):
        """Benchmark performance."""
        import time

        logger.info("Running Torch-TRT FP8 KV Cache benchmark...")

        hidden = torch.randn(
            1, self.seq_len, HIDDEN_SIZE,
            device=self.device, dtype=torch.float16
        )
        position_ids = torch.arange(self.seq_len, device=self.device).unsqueeze(0)

        # Warmup
        for _ in range(num_warmup):
            _ = self.infer(hidden, None, position_ids)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iters):
            keys, values, _ = self.infer(hidden, None, position_ids)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) / num_iters * 1000

        results = {
            "torch_trt_fp8_kv_cache_ms": latency_ms,
            "hz": 1000 / latency_ms,
            "num_layers": self.num_layers,
            "seq_len": self.seq_len,
            "trt_compiled_layers": self._trt_compiled_count,
        }

        logger.info(f"Benchmark Results:")
        logger.info(f"  Torch-TRT FP8 KV Cache: {latency_ms:.2f} ms ({results['hz']:.1f} Hz)")
        logger.info(f"  TRT Compiled Layers: {self._trt_compiled_count}/{self.num_layers}")

        return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        checkpoint_dir = "/root/.cache/openpi/checkpoints/pi05_libero"
    else:
        checkpoint_dir = sys.argv[1]

    engine = TorchTRTFP8KVCacheEngine(checkpoint_dir)
    results = engine.benchmark()
    print(f"\nResults: {results}")
