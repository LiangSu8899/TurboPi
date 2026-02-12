#!/usr/bin/env python3
"""
Test W4A16 TRT Plugin Inference

This script tests the actual TensorRT plugin (C++) inference, NOT the Python TVM interface.

The W4A16MLPPlugin:
1. Loads TVM kernel .so files via C++ (no Python overhead)
2. Executes MLP: gate_proj -> up_proj -> GeLU*up -> down_proj
3. Uses pre-quantized nvFP4 weights

Usage:
    In Docker container (turbo_pi_eval):
    python /workspace/scripts/test_w4a16_trt_plugin_inference.py
"""

import sys
import os
import ctypes
import numpy as np
import time

# Add paths
sys.path.insert(0, "/workspace/src")

# Constants
HIDDEN_SIZE = 2048
MLP_DIM = 16384
BLOCK_SIZE = 32
NUM_LAYERS = 18

# Paths
PLUGIN_LIB = "/workspace/w4a16_tvm_plugin/lib/build/libw4a16_trt_plugin.so"
TVM_KERNEL_LIB = "/workspace/w4a16_tvm_plugin/lib"
QUANTIZED_WEIGHTS = "/workspace/quantized_weights/mlp_weights_nvfp4.safetensors"


def check_dependencies():
    """Check if all required files exist."""
    print("=" * 60)
    print("Checking dependencies...")
    print("=" * 60)

    files = [
        (PLUGIN_LIB, "TRT Plugin"),
        (f"{TVM_KERNEL_LIB}/libw4a16_gate_up.so", "TVM gate_up kernel"),
        (f"{TVM_KERNEL_LIB}/libw4a16_gelu_mul.so", "TVM gelu_mul kernel"),
        (f"{TVM_KERNEL_LIB}/libw4a16_down_proj.so", "TVM down_proj kernel"),
        (QUANTIZED_WEIGHTS, "Quantized weights"),
    ]

    all_ok = True
    for path, name in files:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  [OK] {name}: {path} ({size:,} bytes)")
        else:
            print(f"  [MISSING] {name}: {path}")
            all_ok = False

    return all_ok


def test_trt_plugin_load():
    """Test loading the TRT plugin."""
    print("\n" + "=" * 60)
    print("Step 1: Load TensorRT Plugin")
    print("=" * 60)

    try:
        import tensorrt as trt
        print(f"TensorRT version: {trt.__version__}")
    except ImportError as e:
        print(f"TensorRT not available: {e}")
        return False

    # Load plugin library
    print(f"\nLoading plugin: {PLUGIN_LIB}")
    try:
        ctypes.CDLL(PLUGIN_LIB, mode=ctypes.RTLD_GLOBAL)
        print("  Plugin library loaded successfully")
    except Exception as e:
        print(f"  Failed to load plugin: {e}")
        return False

    # Initialize TensorRT plugins
    trt.init_libnvinfer_plugins(None, "")

    # Check if our plugin is registered
    registry = trt.get_plugin_registry()
    creator = registry.get_creator("W4A16MLPPlugin", "1", "")

    if creator:
        print(f"  W4A16MLPPlugin registered: {creator.name} v{creator.plugin_version}")
        return True
    else:
        print("  W4A16MLPPlugin NOT found in registry")
        # List available plugins
        print("\n  Available plugins:")
        for i in range(registry.num_creators):
            c = registry.get_creator_by_index(i)
            print(f"    - {c.name} v{c.plugin_version}")
        return False


def build_single_mlp_engine():
    """Build a TRT engine with single W4A16 MLP plugin."""
    print("\n" + "=" * 60)
    print("Step 2: Build TRT Engine with W4A16 Plugin")
    print("=" * 60)

    import tensorrt as trt
    from safetensors import safe_open

    # Load quantized weights for layer 0
    print("\nLoading quantized weights...")
    with safe_open(QUANTIZED_WEIGHTS, framework="numpy") as f:
        gate_W = f.get_tensor("layer.0.gate_proj.weight_packed")
        gate_S = f.get_tensor("layer.0.gate_proj.scales")
        up_W = f.get_tensor("layer.0.up_proj.weight_packed")
        up_S = f.get_tensor("layer.0.up_proj.scales")
        down_W = f.get_tensor("layer.0.down_proj.weight_packed")
        down_S = f.get_tensor("layer.0.down_proj.scales")

    print(f"  gate_W: {gate_W.shape} {gate_W.dtype}")
    print(f"  gate_S: {gate_S.shape} {gate_S.dtype}")
    print(f"  down_W: {down_W.shape} {down_W.dtype}")

    # Create TRT builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Create input tensor
    x_input = network.add_input("input", trt.DataType.FLOAT, (1, HIDDEN_SIZE))

    # Add weight tensors as constants
    # Note: TRT requires INT8 for uint8 weights, and we need to use numpy arrays
    gate_W_const = network.add_constant(gate_W.shape, trt.Weights(gate_W.view(np.int8)))
    gate_S_const = network.add_constant(gate_S.shape, trt.Weights(gate_S.astype(np.float32)))
    up_W_const = network.add_constant(up_W.shape, trt.Weights(up_W.view(np.int8)))
    up_S_const = network.add_constant(up_S.shape, trt.Weights(up_S.astype(np.float32)))
    down_W_const = network.add_constant(down_W.shape, trt.Weights(down_W.view(np.int8)))
    down_S_const = network.add_constant(down_S.shape, trt.Weights(down_S.astype(np.float32)))

    # Get plugin creator
    registry = trt.get_plugin_registry()
    creator = registry.get_creator("W4A16MLPPlugin", "1", "")

    if not creator:
        print("ERROR: W4A16MLPPlugin not found!")
        return None

    # Create plugin fields
    fields = trt.PluginFieldCollection([
        trt.PluginField("hidden_size", np.array([HIDDEN_SIZE], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("intermediate_size", np.array([MLP_DIM], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("lib_dir", TVM_KERNEL_LIB.encode(), trt.PluginFieldType.CHAR),
    ])

    # Create plugin
    plugin = creator.create_plugin("w4a16_mlp", fields, trt.TensorRTPhase.BUILD)

    if not plugin:
        print("ERROR: Failed to create W4A16MLPPlugin!")
        return None

    print(f"\nPlugin created successfully")

    # Add plugin to network
    # Inputs: x, gate_W, gate_S, up_W, up_S, down_W, down_S
    plugin_inputs = [
        x_input,
        gate_W_const.get_output(0),
        gate_S_const.get_output(0),
        up_W_const.get_output(0),
        up_S_const.get_output(0),
        down_W_const.get_output(0),
        down_S_const.get_output(0),
    ]

    mlp_layer = network.add_plugin_v3(plugin_inputs, [], plugin)
    mlp_layer.name = "w4a16_mlp_layer"

    # Mark output
    output = mlp_layer.get_output(0)
    output.name = "output"
    network.mark_output(output)

    print(f"\nBuilding TRT engine...")
    engine = builder.build_serialized_network(network, config)

    if engine:
        print(f"  Engine built successfully ({len(engine):,} bytes)")
        return engine
    else:
        print("  Engine build failed!")
        return None


def benchmark_trt_engine(engine_bytes, num_warmup=20, num_iters=100):
    """Benchmark the TRT engine."""
    print("\n" + "=" * 60)
    print("Step 3: Benchmark TRT Engine")
    print("=" * 60)

    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    # Allocate buffers
    device = torch.device("cuda")
    x = torch.randn(1, HIDDEN_SIZE, device=device, dtype=torch.float32)
    output = torch.empty(1, HIDDEN_SIZE, device=device, dtype=torch.float32)

    # Set tensor addresses
    context.set_tensor_address("input", x.data_ptr())
    context.set_tensor_address("output", output.data_ptr())

    # Create CUDA stream
    stream = torch.cuda.Stream()

    # Warmup
    print(f"\nWarming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # Benchmark
    print(f"Benchmarking ({num_iters} iterations)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters * 1000

    print(f"\n  Single MLP layer (TRT Plugin): {elapsed:.4f} ms")
    print(f"  18 layers estimated: {elapsed * 18:.3f} ms")

    return elapsed


def benchmark_comparison():
    """Compare TRT Plugin vs Python TVM interface."""
    print("\n" + "=" * 60)
    print("Step 4: Comparison with Python TVM Interface")
    print("=" * 60)

    # Set up TVM environment
    tvm_paths = [
        "/workspace/external/tvm/build",
        "/workspace/external/tvm/build/lib",
        "/workspace/external/tvm/build/3rdparty/cutlass_fpA_intB_gemm/cutlass_kernels",
        "/workspace/external/tvm/build/3rdparty/libflash_attn/src",
    ]
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(tvm_paths) + ":" + ld_path

    sys.path.insert(0, "/workspace/external/tvm/python")

    try:
        import tvm
        from openpi.models_pytorch.tvm_kernels.w4a16_gemv import (
            create_w4a16_gemv_fast,
            build_kernel,
            quantize_to_nvfp4_packed,
            BLOCK_SIZE,
        )

        H = HIDDEN_SIZE
        I = MLP_DIM
        num_blocks_H = (H + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_blocks_I = (I + BLOCK_SIZE - 1) // BLOCK_SIZE

        device = tvm.runtime.cuda(0)

        # Build kernels
        print("\nBuilding TVM kernels (Python interface)...")
        gate_up_kernel = create_w4a16_gemv_fast(I, H)
        gate_up_mod = build_kernel(gate_up_kernel, target="cuda -arch=sm_110")
        gate_up_func = gate_up_mod["w4a16_gemv_fast"]

        down_kernel = create_w4a16_gemv_fast(H, I)
        down_mod = build_kernel(down_kernel, target="cuda -arch=sm_110")
        down_func = down_mod["w4a16_gemv_fast"]

        # Prepare data
        np.random.seed(42)
        gate_W_np = np.random.randn(I, H).astype(np.float32) * 0.1
        up_W_np = np.random.randn(I, H).astype(np.float32) * 0.1
        down_W_np = np.random.randn(H, I).astype(np.float32) * 0.1

        gate_W_packed, gate_scales = quantize_to_nvfp4_packed(gate_W_np)
        up_W_packed, up_scales = quantize_to_nvfp4_packed(up_W_np)
        down_W_packed, down_scales = quantize_to_nvfp4_packed(down_W_np)

        # TVM arrays
        x_tvm = tvm.runtime.empty((1, H), "float32", device)
        x_np = np.random.randn(1, H).astype(np.float32)
        x_tvm.copyfrom(x_np)

        gate_W_tvm = tvm.runtime.empty((I, H // 2), "uint8", device)
        gate_W_tvm.copyfrom(gate_W_packed)
        gate_scales_tvm = tvm.runtime.empty((I, num_blocks_H), "float32", device)
        gate_scales_tvm.copyfrom(gate_scales)

        up_W_tvm = tvm.runtime.empty((I, H // 2), "uint8", device)
        up_W_tvm.copyfrom(up_W_packed)
        up_scales_tvm = tvm.runtime.empty((I, num_blocks_H), "float32", device)
        up_scales_tvm.copyfrom(up_scales)

        down_W_tvm = tvm.runtime.empty((H, I // 2), "uint8", device)
        down_W_tvm.copyfrom(down_W_packed)
        down_scales_tvm = tvm.runtime.empty((H, num_blocks_I), "float32", device)
        down_scales_tvm.copyfrom(down_scales)

        gate_out_tvm = tvm.runtime.empty((1, I), "float32", device)
        up_out_tvm = tvm.runtime.empty((1, I), "float32", device)
        down_out_tvm = tvm.runtime.empty((1, H), "float32", device)

        # Warmup
        for _ in range(20):
            gate_up_func(x_tvm, gate_W_tvm, gate_scales_tvm, gate_out_tvm)
            gate_up_func(x_tvm, up_W_tvm, up_scales_tvm, up_out_tvm)
            down_func(gate_out_tvm, down_W_tvm, down_scales_tvm, down_out_tvm)
        device.sync()

        # Benchmark
        runs = 100
        device.sync()
        start = time.time()
        for _ in range(runs):
            gate_up_func(x_tvm, gate_W_tvm, gate_scales_tvm, gate_out_tvm)
            gate_up_func(x_tvm, up_W_tvm, up_scales_tvm, up_out_tvm)
            down_func(gate_out_tvm, down_W_tvm, down_scales_tvm, down_out_tvm)
        device.sync()
        python_time = (time.time() - start) / runs * 1000

        print(f"\n  Python TVM (single layer): {python_time:.4f} ms")
        print(f"  Python TVM (18 layers): {python_time * 18:.3f} ms")

        return python_time

    except Exception as e:
        print(f"Python TVM benchmark failed: {e}")
        return None


def main():
    print("=" * 60)
    print("W4A16 TRT Plugin Inference Test")
    print("=" * 60)
    print("This tests the ACTUAL TensorRT Plugin (C++), NOT Python TVM")
    print()

    # Check dependencies
    if not check_dependencies():
        print("\nMissing dependencies. Please build first.")
        return

    # Test plugin loading
    if not test_trt_plugin_load():
        print("\nPlugin loading failed.")
        return

    # Build TRT engine
    engine = build_single_mlp_engine()
    if not engine:
        print("\nEngine build failed.")
        return

    # Benchmark TRT engine
    trt_time = benchmark_trt_engine(engine)

    # Compare with Python TVM
    python_time = benchmark_comparison()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if trt_time and python_time:
        overhead = python_time - trt_time
        print(f"TRT Plugin (C++):    {trt_time:.4f} ms/layer")
        print(f"Python TVM:          {python_time:.4f} ms/layer")
        print(f"Python overhead:     {overhead:.4f} ms/layer")
        print()
        print(f"18 layers:")
        print(f"  TRT Plugin:        {trt_time * 18:.3f} ms")
        print(f"  Python TVM:        {python_time * 18:.3f} ms")
        print(f"  TRT FP8 baseline:  12.39 ms")
        print()
        if trt_time * 18 < 12.39:
            print(f"  W4A16 TRT Plugin is {12.39 - trt_time * 18:.2f}ms FASTER than TRT FP8!")
        else:
            print(f"  W4A16 TRT Plugin is {trt_time * 18 - 12.39:.2f}ms slower than TRT FP8")


if __name__ == "__main__":
    main()
