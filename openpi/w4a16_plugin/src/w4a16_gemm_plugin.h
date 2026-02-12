/*
 * W4A16 GEMM TensorRT Plugin
 *
 * Weight: INT4 (4-bit integer with per-group scaling)
 * Activation: BF16 (16-bit bfloat16)
 * Output: BF16
 *
 * Uses CUTLASS SM100/SM110 Mixed-Dtype Tensor Core MMA.
 * Based on CUTLASS Example 86: blackwell_mixed_dtype_gemm
 *
 * Key advantage: Does NOT require tcgen05.mma.block_scale instruction,
 * making it fully compatible with SM110 (Jetson Thor).
 *
 * Author: Claude Code
 * Date: 2026-02-09
 */

#ifndef W4A16_GEMM_PLUGIN_H
#define W4A16_GEMM_PLUGIN_H

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cstring>
#include <string>
#include <vector>

namespace turbo_pi {

// Plugin configuration constants
constexpr char const* W4A16_GEMM_PLUGIN_NAME = "W4A16GemmPlugin";
constexpr char const* W4A16_GEMM_PLUGIN_VERSION = "1";
constexpr char const* W4A16_GEMM_PLUGIN_NAMESPACE = "";

// Quantization group size (K dimension)
constexpr int W4A16_GROUP_SIZE = 128;  // Scale granularity in K dimension

// Forward declaration of CUDA kernel
extern "C" void w4a16GemmForward(
    const void* activation,        // BF16 [M, K]
    const void* weight_packed,     // INT4 packed as INT8 [N, K/2] (column-major conceptually)
    const void* weight_scale,      // BF16 scales [N, K/GROUP_SIZE]
    const void* weight_zero,       // BF16 zero points [N, K/GROUP_SIZE] (optional, can be nullptr)
    void* output,                  // BF16 [M, N]
    int M,                         // Batch size (number of tokens)
    int N,                         // Output features
    int K,                         // Input features
    float alpha,                   // Scale factor
    float beta,                    // Beta for C accumulation
    void* workspace,
    size_t workspaceSize,
    cudaStream_t stream
);

extern "C" size_t getW4A16GemmWorkspaceSize(
    int M, int N, int K
);

// Weight quantization helper (called during engine build or offline)
extern "C" void quantizeWeightToINT4(
    const __nv_bfloat16* input,    // BF16 [N, K]
    int8_t* output_packed,         // INT4 packed as INT8 [N, K/2]
    __nv_bfloat16* scales,         // BF16 scales [N, K/GROUP_SIZE]
    __nv_bfloat16* zeros,          // BF16 zero points [N, K/GROUP_SIZE]
    int N, int K,
    int group_size,
    cudaStream_t stream
);

// Dequantization helper for verification
extern "C" void dequantizeINT4ToBF16(
    const int8_t* input_packed,    // INT4 packed as INT8 [N, K/2]
    const __nv_bfloat16* scales,   // BF16 scales [N, K/GROUP_SIZE]
    const __nv_bfloat16* zeros,    // BF16 zero points [N, K/GROUP_SIZE]
    __nv_bfloat16* output,         // BF16 [N, K]
    int N, int K,
    int group_size,
    cudaStream_t stream
);

/**
 * W4A16 GEMM Plugin using IPluginV3 interface
 *
 * This plugin implements mixed-precision GEMM with:
 * - 4-bit INT4 weights (pre-quantized and packed, with per-group scales)
 * - 16-bit BF16 activations (native format, no runtime quantization)
 * - BF16 output
 *
 * Input tensors:
 *   0: activation [M, K] - BF16
 *   1: weight_packed [N, K/2] - INT8 (packed INT4)
 *   2: weight_scale [N, K/GROUP_SIZE] - BF16 scales
 *   3: weight_zero [N, K/GROUP_SIZE] - BF16 zero points (optional)
 *
 * Output tensor:
 *   0: output [M, N] - BF16
 */
class W4A16GemmPlugin : public nvinfer1::IPluginV3,
                         public nvinfer1::IPluginV3OneCore,
                         public nvinfer1::IPluginV3OneBuild,
                         public nvinfer1::IPluginV3OneRuntime {
public:
    W4A16GemmPlugin(int inFeatures, int outFeatures, int groupSize = W4A16_GROUP_SIZE,
                    bool useZeroPoint = true, float alpha = 1.0f, float beta = 0.0f);
    W4A16GemmPlugin(const W4A16GemmPlugin& other);
    ~W4A16GemmPlugin() override = default;

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
    int mInFeatures;     // K dimension
    int mOutFeatures;    // N dimension
    int mGroupSize;      // Scale group size
    bool mUseZeroPoint;  // Whether to use zero point
    float mAlpha;        // Output scale
    float mBeta;         // Beta for C accumulation

    // Runtime dimensions
    int mBatchSize{0};   // M dimension (dynamic)

    // Serialization fields
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

/**
 * Plugin Creator for W4A16 GEMM
 */
class W4A16GemmPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    W4A16GemmPluginCreator();
    ~W4A16GemmPluginCreator() override = default;

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

#endif  // W4A16_GEMM_PLUGIN_H
