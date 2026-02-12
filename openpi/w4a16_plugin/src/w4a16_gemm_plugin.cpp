/*
 * W4A16 GEMM TensorRT Plugin Implementation
 *
 * Uses IPluginV3 interface for TensorRT 10.x compatibility.
 */

#include "w4a16_gemm_plugin.h"
#include <cstring>
#include <iostream>
#include <cassert>

namespace turbo_pi {

// Static member initialization
nvinfer1::PluginFieldCollection W4A16GemmPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> W4A16GemmPluginCreator::mPluginAttributes;

// ============================================================================
// W4A16GemmPlugin Implementation
// ============================================================================

W4A16GemmPlugin::W4A16GemmPlugin(int inFeatures, int outFeatures, int groupSize,
                                   bool useZeroPoint, float alpha, float beta)
    : mInFeatures(inFeatures)
    , mOutFeatures(outFeatures)
    , mGroupSize(groupSize)
    , mUseZeroPoint(useZeroPoint)
    , mAlpha(alpha)
    , mBeta(beta) {
}

W4A16GemmPlugin::W4A16GemmPlugin(const W4A16GemmPlugin& other)
    : mInFeatures(other.mInFeatures)
    , mOutFeatures(other.mOutFeatures)
    , mGroupSize(other.mGroupSize)
    , mUseZeroPoint(other.mUseZeroPoint)
    , mAlpha(other.mAlpha)
    , mBeta(other.mBeta)
    , mBatchSize(other.mBatchSize) {
}

nvinfer1::IPluginCapability* W4A16GemmPlugin::getCapabilityInterface(
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

nvinfer1::IPluginV3* W4A16GemmPlugin::clone() noexcept {
    try {
        auto* plugin = new W4A16GemmPlugin(*this);
        return plugin;
    } catch (...) {
        return nullptr;
    }
}

char const* W4A16GemmPlugin::getPluginName() const noexcept {
    return W4A16_GEMM_PLUGIN_NAME;
}

char const* W4A16GemmPlugin::getPluginVersion() const noexcept {
    return W4A16_GEMM_PLUGIN_VERSION;
}

char const* W4A16GemmPlugin::getPluginNamespace() const noexcept {
    return W4A16_GEMM_PLUGIN_NAMESPACE;
}

int32_t W4A16GemmPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
    // Validate input/output counts
    // Input 0: activation [M, K] - BF16
    // Input 1: weight_packed [N, K/2] - INT8 (packed INT4)
    // Input 2: weight_scale [N, K/GROUP_SIZE] - BF16
    // Input 3: weight_zero [N, K/GROUP_SIZE] - BF16 (optional)
    // Output 0: [M, N] - BF16
    return 0;
}

int32_t W4A16GemmPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
    // Output is BF16
    outputTypes[0] = nvinfer1::DataType::kBF16;
    return 0;
}

int32_t W4A16GemmPlugin::getOutputShapes(
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

bool W4A16GemmPlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    // pos 0: activation - BF16
    // pos 1: weight_packed - INT8 (packed INT4)
    // pos 2: weight_scale - BF16
    // pos 3: weight_zero - BF16 (optional, only if mUseZeroPoint)
    // pos 3/4: output - BF16

    auto const& desc = inOut[pos].desc;
    int outputPos = mUseZeroPoint ? 4 : 3;

    if (pos == 0) {
        // Activation: BF16
        return desc.type == nvinfer1::DataType::kBF16 &&
               desc.format == nvinfer1::TensorFormat::kLINEAR;
    } else if (pos == 1) {
        // Weight packed: INT8 (2 INT4 values per byte)
        return desc.type == nvinfer1::DataType::kINT8 &&
               desc.format == nvinfer1::TensorFormat::kLINEAR;
    } else if (pos == 2) {
        // Weight scale: BF16
        return desc.type == nvinfer1::DataType::kBF16 &&
               desc.format == nvinfer1::TensorFormat::kLINEAR;
    } else if (pos == 3 && mUseZeroPoint) {
        // Weight zero: BF16
        return desc.type == nvinfer1::DataType::kBF16 &&
               desc.format == nvinfer1::TensorFormat::kLINEAR;
    } else if (pos == outputPos) {
        // Output: BF16
        return desc.type == nvinfer1::DataType::kBF16 &&
               desc.format == nvinfer1::TensorFormat::kLINEAR;
    }
    return false;
}

size_t W4A16GemmPlugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept {
    int M = inputs[0].max.d[0];
    int K = mInFeatures;
    int N = mOutFeatures;

    return getW4A16GemmWorkspaceSize(M, N, K);
}

int32_t W4A16GemmPlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
    mBatchSize = in[0].dims.d[0];
    return 0;
}

int32_t W4A16GemmPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {

    int M = inputDesc[0].dims.d[0];  // Batch size
    int K = mInFeatures;
    int N = mOutFeatures;

    // inputs[0]: BF16 activation [M, K]
    // inputs[1]: INT8 weight_packed [N, K/2]
    // inputs[2]: BF16 weight_scale [N, K/GROUP_SIZE]
    // inputs[3]: BF16 weight_zero [N, K/GROUP_SIZE] (optional)

    const void* weight_zero = mUseZeroPoint ? inputs[3] : nullptr;

    size_t cutlassWorkspaceSize = getW4A16GemmWorkspaceSize(M, N, K);

    w4a16GemmForward(
        inputs[0],           // BF16 activation [M, K]
        inputs[1],           // INT4 weight [N, K/2]
        inputs[2],           // BF16 scales [N, K/GROUP_SIZE]
        weight_zero,         // BF16 zeros [N, K/GROUP_SIZE] or nullptr
        outputs[0],          // BF16 output [M, N]
        M, N, K,
        mAlpha,
        mBeta,
        workspace,
        cutlassWorkspaceSize,
        stream
    );

    return 0;
}

nvinfer1::IPluginV3* W4A16GemmPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* context) noexcept {
    return clone();
}

nvinfer1::PluginFieldCollection const* W4A16GemmPlugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("in_features", &mInFeatures, nvinfer1::PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("out_features", &mOutFeatures, nvinfer1::PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("group_size", &mGroupSize, nvinfer1::PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("use_zero_point", &mUseZeroPoint, nvinfer1::PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("alpha", &mAlpha, nvinfer1::PluginFieldType::kFLOAT32, 1);
    mDataToSerialize.emplace_back("beta", &mBeta, nvinfer1::PluginFieldType::kFLOAT32, 1);

    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ============================================================================
// W4A16GemmPluginCreator Implementation
// ============================================================================

W4A16GemmPluginCreator::W4A16GemmPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("in_features", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("out_features", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("group_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("use_zero_point", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("alpha", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("beta", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

char const* W4A16GemmPluginCreator::getPluginName() const noexcept {
    return W4A16_GEMM_PLUGIN_NAME;
}

char const* W4A16GemmPluginCreator::getPluginVersion() const noexcept {
    return W4A16_GEMM_PLUGIN_VERSION;
}

char const* W4A16GemmPluginCreator::getPluginNamespace() const noexcept {
    return W4A16_GEMM_PLUGIN_NAMESPACE;
}

nvinfer1::PluginFieldCollection const* W4A16GemmPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

nvinfer1::IPluginV3* W4A16GemmPluginCreator::createPlugin(
    char const* name,
    nvinfer1::PluginFieldCollection const* fc,
    nvinfer1::TensorRTPhase phase) noexcept {
    try {
        int inFeatures = 0;
        int outFeatures = 0;
        int groupSize = W4A16_GROUP_SIZE;
        int useZeroPoint = 1;
        float alpha = 1.0f;
        float beta = 0.0f;

        for (int32_t i = 0; i < fc->nbFields; ++i) {
            auto const& field = fc->fields[i];
            if (std::strcmp(field.name, "in_features") == 0) {
                inFeatures = *static_cast<int32_t const*>(field.data);
            } else if (std::strcmp(field.name, "out_features") == 0) {
                outFeatures = *static_cast<int32_t const*>(field.data);
            } else if (std::strcmp(field.name, "group_size") == 0) {
                groupSize = *static_cast<int32_t const*>(field.data);
            } else if (std::strcmp(field.name, "use_zero_point") == 0) {
                useZeroPoint = *static_cast<int32_t const*>(field.data);
            } else if (std::strcmp(field.name, "alpha") == 0) {
                alpha = *static_cast<float const*>(field.data);
            } else if (std::strcmp(field.name, "beta") == 0) {
                beta = *static_cast<float const*>(field.data);
            }
        }

        return new W4A16GemmPlugin(inFeatures, outFeatures, groupSize, useZeroPoint != 0, alpha, beta);
    } catch (...) {
        return nullptr;
    }
}

// Plugin registration
REGISTER_TENSORRT_PLUGIN(W4A16GemmPluginCreator);

}  // namespace turbo_pi
