/*
 * W4A16 MLP TensorRT Plugin
 *
 * Wraps TVM-compiled W4A16 kernels for TensorRT integration.
 *
 * Weight: NVFP4 (4-bit, E2M1 format with block scaling)
 * Activation: FP32 (no quantization loss)
 * Output: FP32
 *
 * Fused operations:
 * 1. gate_proj: [1, H] @ [I, H].T -> [1, I]
 * 2. up_proj:   [1, H] @ [I, H].T -> [1, I]
 * 3. GeLU(gate) * up -> [1, I]
 * 4. down_proj: [1, I] @ [H, I].T -> [1, H]
 *
 * Author: Claude Code
 * Date: 2026-02-11
 */

#ifndef W4A16_TRT_PLUGIN_H
#define W4A16_TRT_PLUGIN_H

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <string>
#include <vector>
#include <memory>

namespace turbo_pi {

// Forward declaration
class W4A16TVMKernels;

// Plugin configuration
constexpr char const* W4A16_MLP_PLUGIN_NAME = "W4A16MLPPlugin";
constexpr char const* W4A16_MLP_PLUGIN_VERSION = "1";
constexpr char const* W4A16_MLP_PLUGIN_NAMESPACE = "";

// Default dimensions (PaliGemma 2B)
constexpr int DEFAULT_HIDDEN_SIZE = 2048;
constexpr int DEFAULT_INTERMEDIATE_SIZE = 16384;
constexpr int DEFAULT_BLOCK_SIZE = 32;

/**
 * W4A16 MLP Plugin using IPluginV3 interface
 *
 * Input tensors:
 *   0: x [1, hidden_size] - FP32 input activation
 *   1: gate_W_packed [intermediate_size, hidden_size/2] - NVFP4 packed
 *   2: gate_scales [intermediate_size, num_blocks_h] - FP32
 *   3: up_W_packed [intermediate_size, hidden_size/2] - NVFP4 packed
 *   4: up_scales [intermediate_size, num_blocks_h] - FP32
 *   5: down_W_packed [hidden_size, intermediate_size/2] - NVFP4 packed
 *   6: down_scales [hidden_size, num_blocks_i] - FP32
 *
 * Output tensor:
 *   0: output [1, hidden_size] - FP32
 */
class W4A16MLPPlugin : public nvinfer1::IPluginV3,
                        public nvinfer1::IPluginV3OneCore,
                        public nvinfer1::IPluginV3OneBuild,
                        public nvinfer1::IPluginV3OneRuntime {
public:
    W4A16MLPPlugin(int hiddenSize, int intermediateSize, const std::string& libDir);
    W4A16MLPPlugin(const W4A16MLPPlugin& other);
    ~W4A16MLPPlugin() override;

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
    int mHiddenSize;
    int mIntermediateSize;
    std::string mLibDir;

    // TVM kernels (lazy initialized)
    std::unique_ptr<W4A16TVMKernels> mKernels;

    // Serialization fields
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

/**
 * Plugin Creator for W4A16 MLP
 */
class W4A16MLPPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    W4A16MLPPluginCreator();
    ~W4A16MLPPluginCreator() override = default;

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

// Plugin registration
extern "C" {
    bool initLibW4A16MLPPlugin();
}

}  // namespace turbo_pi

#endif  // W4A16_TRT_PLUGIN_H
