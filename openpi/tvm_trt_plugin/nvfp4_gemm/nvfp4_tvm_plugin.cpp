/*
 * nvFP4 GEMM TensorRT Plugin - TVM Generated
 *
 * This plugin uses TVM-generated CUDA kernels to implement nvFP4 GEMM
 * without depending on CUTLASS or special Tensor Core instructions.
 *
 * Advantages:
 * - Works on SM110 (Jetson Thor) without mxf8f6f4 limitations
 * - Pre-quantized weights, no runtime quantization bandwidth
 * - Can be used in TensorRT graph for fusion
 *
 * Author: Claude Code
 * Date: 2026-02-10
 */

#include "nvfp4_gemm_tvm_plugin.h"
#include <cuda_runtime.h>
#include <cstring>
#include <string>
#include <vector>

namespace turbo_pi {

constexpr char const* NVFP4_TVM_PLUGIN_NAME = "NVFP4GemmTVMPlugin";
constexpr char const* NVFP4_TVM_PLUGIN_VERSION = "1";
constexpr char const* NVFP4_TVM_PLUGIN_NAMESPACE = "";

// Plugin Fields
static std::vector<nvinfer1::PluginField> sPluginAttributes;
static nvinfer1::PluginFieldCollection sFC;

//==============================================================================
// Plugin Implementation
//==============================================================================

NVFP4GemmTVMPlugin::NVFP4GemmTVMPlugin(int inFeatures, int outFeatures, int blockSize)
    : mInFeatures(inFeatures)
    , mOutFeatures(outFeatures)
    , mBlockSize(blockSize)
    , mBatchSize(0) {
}

NVFP4GemmTVMPlugin::NVFP4GemmTVMPlugin(const NVFP4GemmTVMPlugin& other)
    : mInFeatures(other.mInFeatures)
    , mOutFeatures(other.mOutFeatures)
    , mBlockSize(other.mBlockSize)
    , mBatchSize(other.mBatchSize) {
}

nvinfer1::IPluginCapability* NVFP4GemmTVMPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept {
    try {
        if (type == nvinfer1::PluginCapabilityType::kBUILD) {
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        }
        if (type == nvinfer1::PluginCapabilityType::kRUNTIME) {
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        }
        if (type == nvinfer1::PluginCapabilityType::kCORE) {
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        }
    } catch (...) {
    }
    return nullptr;
}

nvinfer1::IPluginV3* NVFP4GemmTVMPlugin::clone() noexcept {
    try {
        auto* plugin = new NVFP4GemmTVMPlugin(*this);
        return plugin;
    } catch (...) {
        return nullptr;
    }
}

char const* NVFP4GemmTVMPlugin::getPluginName() const noexcept {
    return NVFP4_TVM_PLUGIN_NAME;
}

char const* NVFP4GemmTVMPlugin::getPluginVersion() const noexcept {
    return NVFP4_TVM_PLUGIN_VERSION;
}

char const* NVFP4GemmTVMPlugin::getPluginNamespace() const noexcept {
    return NVFP4_TVM_PLUGIN_NAMESPACE;
}

int32_t NVFP4GemmTVMPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
    return 0;
}

int32_t NVFP4GemmTVMPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
    // Output is float32
    outputTypes[0] = nvinfer1::DataType::kFLOAT;
    return 0;
}

int32_t NVFP4GemmTVMPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::DimsExprs const* shapeInputs,
    int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    // Input: A [M, K], W [N, K], scale_A [M, num_blocks], scale_W [N, num_blocks]
    // Output: C [M, N]
    outputs[0].nbDims = 2;
    outputs[0].d[0] = inputs[0].d[0];  // M (batch)
    outputs[0].d[1] = exprBuilder.constant(mOutFeatures);  // N
    return 0;
}

bool NVFP4GemmTVMPlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    // All inputs and outputs are float32 linear
    return inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].desc.type == nvinfer1::DataType::kFLOAT;
}

size_t NVFP4GemmTVMPlugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept {
    return 0;  // TVM kernel doesn't need workspace
}

int32_t NVFP4GemmTVMPlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
    mBatchSize = in[0].dims.d[0];
    return 0;
}

int32_t NVFP4GemmTVMPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
    // Get dimensions
    int M = inputDesc[0].dims.d[0];
    int K = mInFeatures;
    int N = mOutFeatures;

    // Get pointers
    const float* A = static_cast<const float*>(inputs[0]);        // [M, K]
    const float* W = static_cast<const float*>(inputs[1]);        // [N, K]
    const float* scale_A = static_cast<const float*>(inputs[2]);  // [M, num_blocks]
    const float* scale_W = static_cast<const float*>(inputs[3]);  // [N, num_blocks]
    float* C = static_cast<float*>(outputs[0]);                   // [M, N]

    // Launch TVM-generated kernel
    launch_nvfp4_gemm(A, W, scale_A, scale_W, C, M, N, K, stream);

    return 0;
}

nvinfer1::IPluginV3* NVFP4GemmTVMPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* context) noexcept {
    return clone();
}

nvinfer1::PluginFieldCollection const* NVFP4GemmTVMPlugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("in_features", &mInFeatures, nvinfer1::PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("out_features", &mOutFeatures, nvinfer1::PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("block_size", &mBlockSize, nvinfer1::PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

//==============================================================================
// Plugin Creator Implementation
//==============================================================================

NVFP4GemmTVMPluginCreator::NVFP4GemmTVMPluginCreator() {
    sPluginAttributes.clear();
    sPluginAttributes.emplace_back(nvinfer1::PluginField("in_features", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    sPluginAttributes.emplace_back(nvinfer1::PluginField("out_features", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    sPluginAttributes.emplace_back(nvinfer1::PluginField("block_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    sFC.nbFields = sPluginAttributes.size();
    sFC.fields = sPluginAttributes.data();
}

char const* NVFP4GemmTVMPluginCreator::getPluginName() const noexcept {
    return NVFP4_TVM_PLUGIN_NAME;
}

char const* NVFP4GemmTVMPluginCreator::getPluginVersion() const noexcept {
    return NVFP4_TVM_PLUGIN_VERSION;
}

char const* NVFP4GemmTVMPluginCreator::getPluginNamespace() const noexcept {
    return NVFP4_TVM_PLUGIN_NAMESPACE;
}

nvinfer1::PluginFieldCollection const* NVFP4GemmTVMPluginCreator::getFieldNames() noexcept {
    return &sFC;
}

nvinfer1::IPluginV3* NVFP4GemmTVMPluginCreator::createPlugin(
    char const* name,
    nvinfer1::PluginFieldCollection const* fc,
    nvinfer1::TensorRTPhase phase) noexcept {
    int inFeatures = 0;
    int outFeatures = 0;
    int blockSize = 32;

    for (int i = 0; i < fc->nbFields; ++i) {
        const char* attrName = fc->fields[i].name;
        if (!strcmp(attrName, "in_features")) {
            inFeatures = *static_cast<const int*>(fc->fields[i].data);
        } else if (!strcmp(attrName, "out_features")) {
            outFeatures = *static_cast<const int*>(fc->fields[i].data);
        } else if (!strcmp(attrName, "block_size")) {
            blockSize = *static_cast<const int*>(fc->fields[i].data);
        }
    }

    try {
        auto* plugin = new NVFP4GemmTVMPlugin(inFeatures, outFeatures, blockSize);
        return plugin;
    } catch (...) {
        return nullptr;
    }
}

// Register plugin
REGISTER_TENSORRT_PLUGIN(NVFP4GemmTVMPluginCreator);

}  // namespace turbo_pi
