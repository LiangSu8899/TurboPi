/*
 * W4A8 GEMM TensorRT Plugin Implementation
 *
 * Uses IPluginV3 interface for TensorRT 10.x compatibility.
 */

#include "w4a8_gemm_plugin.h"
#include <cstring>
#include <iostream>
#include <cassert>

namespace turbo_pi {

// Static member initialization
nvinfer1::PluginFieldCollection W4A8GemmPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> W4A8GemmPluginCreator::mPluginAttributes;

// ============================================================================
// W4A8GemmPlugin Implementation
// ============================================================================

W4A8GemmPlugin::W4A8GemmPlugin(int inFeatures, int outFeatures, float alpha)
    : mInFeatures(inFeatures)
    , mOutFeatures(outFeatures)
    , mAlpha(alpha) {
}

W4A8GemmPlugin::W4A8GemmPlugin(const W4A8GemmPlugin& other)
    : mInFeatures(other.mInFeatures)
    , mOutFeatures(other.mOutFeatures)
    , mAlpha(other.mAlpha)
    , mBatchSize(other.mBatchSize) {
}

nvinfer1::IPluginCapability* W4A8GemmPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept {
    try {
        if (type == nvinfer1::PluginCapabilityType::kBUILD) {
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        }
        if (type == nvinfer1::PluginCapabilityType::kRUNTIME) {
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        }
        assert(type == nvinfer1::PluginCapabilityType::kCORE);
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    } catch (...) {
        return nullptr;
    }
}

nvinfer1::IPluginV3* W4A8GemmPlugin::clone() noexcept {
    try {
        auto* plugin = new W4A8GemmPlugin(*this);
        return plugin;
    } catch (...) {
        return nullptr;
    }
}

char const* W4A8GemmPlugin::getPluginName() const noexcept {
    return W4A8_GEMM_PLUGIN_NAME;
}

char const* W4A8GemmPlugin::getPluginVersion() const noexcept {
    return W4A8_GEMM_PLUGIN_VERSION;
}

char const* W4A8GemmPlugin::getPluginNamespace() const noexcept {
    return W4A8_GEMM_PLUGIN_NAMESPACE;
}

int32_t W4A8GemmPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
    // Validate input/output counts
    // Input 0: activation [M, K] - BF16
    // Input 1: weight_packed [K/2, N] - INT8 (packed NVFP4)
    // Input 2: weight_scale [K/BLOCK_SIZE, N] - FP8
    // Output 0: [M, N] - BF16
    return 0;
}

int32_t W4A8GemmPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
    // Output is BF16
    outputTypes[0] = nvinfer1::DataType::kBF16;
    return 0;
}

int32_t W4A8GemmPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::DimsExprs const* shapeInputs,
    int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    // Input 0: activation [M, K]
    // Output 0: [M, N]
    outputs[0].nbDims = 2;
    outputs[0].d[0] = inputs[0].d[0];  // M (batch dimension)
    outputs[0].d[1] = exprBuilder.constant(mOutFeatures);  // N
    return 0;
}

bool W4A8GemmPlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    // pos 0: activation - BF16
    // pos 1: weight_packed - INT8 (packed NVFP4)
    // pos 2: weight_scale - FP8
    // pos 3: output - BF16

    auto const& desc = inOut[pos].desc;

    if (pos == 0) {
        // Activation: BF16
        return desc.type == nvinfer1::DataType::kBF16 &&
               desc.format == nvinfer1::TensorFormat::kLINEAR;
    } else if (pos == 1) {
        // Weight packed: INT8 (2 NVFP4 values per byte)
        return desc.type == nvinfer1::DataType::kINT8 &&
               desc.format == nvinfer1::TensorFormat::kLINEAR;
    } else if (pos == 2) {
        // Weight scale: FP8
        return desc.type == nvinfer1::DataType::kFP8 &&
               desc.format == nvinfer1::TensorFormat::kLINEAR;
    } else if (pos == 3) {
        // Output: BF16
        return desc.type == nvinfer1::DataType::kBF16 &&
               desc.format == nvinfer1::TensorFormat::kLINEAR;
    }
    return false;
}

size_t W4A8GemmPlugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept {
    // Workspace for FP8 quantized activation and scales
    int M = inputs[0].max.d[0];
    int K = mInFeatures;

    size_t activationFP8Size = M * K * sizeof(__nv_fp8_e4m3);
    size_t activationScaleSize = M * (K / NVFP4_BLOCK_SIZE) * sizeof(__nv_fp8_e4m3);
    size_t cutlassWorkspace = getW4A8GemmWorkspaceSize(M, mOutFeatures, K);

    return activationFP8Size + activationScaleSize + cutlassWorkspace;
}

int32_t W4A8GemmPlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
    mBatchSize = in[0].dims.d[0];
    return 0;
}

int32_t W4A8GemmPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {

    int M = inputDesc[0].dims.d[0];  // Batch size
    int K = mInFeatures;
    int N = mOutFeatures;

    // Workspace layout:
    // [0, activationFP8Size): FP8 quantized activation
    // [activationFP8Size, activationFP8Size + scaleSize): FP8 scales
    // [activationFP8Size + scaleSize, ...): CUTLASS workspace

    size_t activationFP8Size = M * K * sizeof(__nv_fp8_e4m3);
    size_t scaleSize = M * (K / NVFP4_BLOCK_SIZE) * sizeof(__nv_fp8_e4m3);

    void* activationFP8 = workspace;
    void* activationScale = static_cast<char*>(workspace) + activationFP8Size;
    void* cutlassWorkspace = static_cast<char*>(workspace) + activationFP8Size + scaleSize;

    // Step 1: Quantize BF16 activation to FP8 with per-row scaling
    quantizeActivationToFP8(
        static_cast<const __nv_bfloat16*>(inputs[0]),  // BF16 activation
        static_cast<__nv_fp8_e4m3*>(activationFP8),    // FP8 output
        static_cast<__nv_fp8_e4m3*>(activationScale),  // FP8 scales
        M, K,
        stream
    );

    // Step 2: CUTLASS W4A8 GEMM
    // inputs[1]: weight_packed (NVFP4)
    // inputs[2]: weight_scale (FP8)
    size_t cutlassWorkspaceSize = getW4A8GemmWorkspaceSize(M, N, K);

    w4a8GemmForward(
        activationFP8,           // FP8 activation [M, K]
        activationScale,         // FP8 scales [M, K/BLOCK_SIZE]
        inputs[1],               // NVFP4 weight [K/2, N]
        inputs[2],               // FP8 weight scales [K/BLOCK_SIZE, N]
        outputs[0],              // BF16 output [M, N]
        M, N, K,
        mAlpha,
        cutlassWorkspace,
        cutlassWorkspaceSize,
        stream
    );

    return 0;
}

nvinfer1::IPluginV3* W4A8GemmPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* context) noexcept {
    return clone();
}

nvinfer1::PluginFieldCollection const* W4A8GemmPlugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("in_features", &mInFeatures, nvinfer1::PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("out_features", &mOutFeatures, nvinfer1::PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("alpha", &mAlpha, nvinfer1::PluginFieldType::kFLOAT32, 1);

    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ============================================================================
// W4A8GemmPluginCreator Implementation
// ============================================================================

W4A8GemmPluginCreator::W4A8GemmPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("in_features", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("out_features", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("alpha", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

char const* W4A8GemmPluginCreator::getPluginName() const noexcept {
    return W4A8_GEMM_PLUGIN_NAME;
}

char const* W4A8GemmPluginCreator::getPluginVersion() const noexcept {
    return W4A8_GEMM_PLUGIN_VERSION;
}

char const* W4A8GemmPluginCreator::getPluginNamespace() const noexcept {
    return W4A8_GEMM_PLUGIN_NAMESPACE;
}

nvinfer1::PluginFieldCollection const* W4A8GemmPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

nvinfer1::IPluginV3* W4A8GemmPluginCreator::createPlugin(
    char const* name,
    nvinfer1::PluginFieldCollection const* fc,
    nvinfer1::TensorRTPhase phase) noexcept {
    try {
        int inFeatures = 0;
        int outFeatures = 0;
        float alpha = 1.0f;

        for (int32_t i = 0; i < fc->nbFields; ++i) {
            auto const& field = fc->fields[i];
            if (std::strcmp(field.name, "in_features") == 0) {
                inFeatures = *static_cast<int32_t const*>(field.data);
            } else if (std::strcmp(field.name, "out_features") == 0) {
                outFeatures = *static_cast<int32_t const*>(field.data);
            } else if (std::strcmp(field.name, "alpha") == 0) {
                alpha = *static_cast<float const*>(field.data);
            }
        }

        return new W4A8GemmPlugin(inFeatures, outFeatures, alpha);
    } catch (...) {
        return nullptr;
    }
}

// Plugin registration
REGISTER_TENSORRT_PLUGIN(W4A8GemmPluginCreator);

}  // namespace turbo_pi
