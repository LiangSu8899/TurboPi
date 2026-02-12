/*
 * W4A8 GEMM TensorRT Plugin
 *
 * Weight: NVFP4 (4-bit, E2M1 format with block scaling)
 * Activation: FP8 (8-bit, E4M3 format with per-row scaling)
 * Output: BF16
 *
 * Uses CUTLASS SM100 Block Scaled Tensor Core MMA for maximum throughput.
 * Based on CUTLASS Example 72c: blackwell_mixed_mxfp8_bf16_gemm
 *
 * Author: Claude Code
 * Date: 2026-02-09
 */

#ifndef W4A8_GEMM_PLUGIN_H
#define W4A8_GEMM_PLUGIN_H

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cstring>
#include <string>
#include <vector>

namespace turbo_pi {

// Plugin configuration constants
constexpr char const* W4A8_GEMM_PLUGIN_NAME = "W4A8GemmPlugin";
constexpr char const* W4A8_GEMM_PLUGIN_VERSION = "1";
constexpr char const* W4A8_GEMM_PLUGIN_NAMESPACE = "";

// NVFP4 block size (must match CUTLASS configuration)
constexpr int NVFP4_BLOCK_SIZE = 16;  // MX format uses 16-element blocks

// Forward declaration of CUDA kernel
extern "C" void w4a8GemmForward(
    const void* activation,        // FP8 E4M3 [M, K]
    const void* activation_scale,  // FP8 E4M3 scales [M, K/BLOCK_SIZE]
    const void* weight_packed,     // NVFP4 packed [K/2, N]
    const void* weight_scale,      // FP8 E4M3 scales [K/BLOCK_SIZE, N]
    void* output,                  // BF16 [M, N]
    int M,                         // Batch size (number of tokens)
    int N,                         // Output features
    int K,                         // Input features
    float alpha,                   // Scale factor
    void* workspace,
    size_t workspaceSize,
    cudaStream_t stream
);

extern "C" size_t getW4A8GemmWorkspaceSize(
    int M, int N, int K
);

// Quantization helpers (called during engine build)
extern "C" void quantizeActivationToFP8(
    const __nv_bfloat16* input,    // BF16 [M, K]
    __nv_fp8_e4m3* output,         // FP8 [M, K]
    __nv_fp8_e4m3* scales,         // FP8 scales [M, K/BLOCK_SIZE]
    int M, int K,
    cudaStream_t stream
);

extern "C" void quantizeWeightToNVFP4(
    const __nv_bfloat16* input,    // BF16 [N, K] (transposed)
    uint8_t* output_packed,        // NVFP4 packed [K/2, N]
    __nv_fp8_e4m3* scales,         // FP8 scales [K/BLOCK_SIZE, N]
    int N, int K,
    cudaStream_t stream
);

/**
 * W4A8 GEMM Plugin using IPluginV3 interface
 *
 * This plugin implements mixed-precision GEMM with:
 * - 4-bit NVFP4 weights (pre-quantized and packed)
 * - 8-bit FP8 activations (dynamically quantized at runtime)
 * - BF16 output
 *
 * Input tensors:
 *   0: activation [M, K] - BF16 (will be quantized to FP8 internally)
 *   1: weight_packed [K/2, N] - NVFP4 packed (pre-quantized)
 *   2: weight_scale [K/BLOCK_SIZE, N] - FP8 scales
 *
 * Output tensor:
 *   0: output [M, N] - BF16
 */
class W4A8GemmPlugin : public nvinfer1::IPluginV3,
                        public nvinfer1::IPluginV3OneCore,
                        public nvinfer1::IPluginV3OneBuild,
                        public nvinfer1::IPluginV3OneRuntime {
public:
    W4A8GemmPlugin(int inFeatures, int outFeatures, float alpha = 1.0f);
    W4A8GemmPlugin(const W4A8GemmPlugin& other);
    ~W4A8GemmPlugin() override = default;

    // IPluginV3 core methods
    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override;

    nvinfer1::IPluginV3* clone() noexcept override;

    // IPluginV3OneCore methods
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV3OneBuild methods
    int32_t configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in,
        int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    int32_t getOutputDataTypes(
        nvinfer1::DataType* outputTypes,
        int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(
        nvinfer1::DimsExprs const* inputs,
        int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs,
        int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int32_t pos,
        nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs,
        int32_t nbOutputs) noexcept override;

    int32_t getNbOutputs() const noexcept override { return 1; }

    size_t getWorkspaceSize(
        nvinfer1::DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;

    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override {
        return 0;
    }

    int32_t getNbTactics() noexcept override { return 0; }

    int32_t getFormatCombinationLimit() noexcept override { return 1; }

    char const* getMetadataString() noexcept override { return nullptr; }

    // IPluginV3OneRuntime methods
    int32_t onShapeChange(
        nvinfer1::PluginTensorDesc const* in,
        int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override;

    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext* context) noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    int mInFeatures;   // K dimension
    int mOutFeatures;  // N dimension
    float mAlpha;      // Output scale

    // Runtime dimensions
    int mBatchSize{0}; // M dimension (dynamic)

    // Workspace for FP8 quantized activation
    mutable void* mActivationFP8{nullptr};
    mutable void* mActivationScale{nullptr};

    // Serialization fields
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

/**
 * Plugin Creator for W4A8 GEMM
 */
class W4A8GemmPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    W4A8GemmPluginCreator();
    ~W4A8GemmPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV3* createPlugin(
        char const* name,
        nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

}  // namespace turbo_pi

#endif  // W4A8_GEMM_PLUGIN_H
