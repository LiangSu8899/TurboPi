/**
 * NVFP4 Packed GEMV TensorRT Plugin Implementation
 */

#include "nvfp4_packed_plugin.h"
#include <cstring>
#include <iostream>
#include <cassert>

namespace turbo_pi {

// Static member initialization
nvinfer1::PluginFieldCollection NVFP4PackedPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> NVFP4PackedPluginCreator::mPluginAttributes;

// ============================================================================
// NVFP4PackedPlugin Implementation
// ============================================================================

NVFP4PackedPlugin::NVFP4PackedPlugin(int inFeatures, int outFeatures,
                                       ActivationType activationType, bool hasBias)
    : mInFeatures(inFeatures)
    , mOutFeatures(outFeatures)
    , mActivationType(activationType)
    , mHasBias(hasBias) {
}

NVFP4PackedPlugin::NVFP4PackedPlugin(const NVFP4PackedPlugin& other)
    : mInFeatures(other.mInFeatures)
    , mOutFeatures(other.mOutFeatures)
    , mActivationType(other.mActivationType)
    , mHasBias(other.mHasBias)
    , mBatchSize(other.mBatchSize)
    , mDataType(other.mDataType) {
}

nvinfer1::IPluginCapability* NVFP4PackedPlugin::getCapabilityInterface(
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

nvinfer1::IPluginV3* NVFP4PackedPlugin::clone() noexcept {
    try {
        return new NVFP4PackedPlugin(*this);
    } catch (...) {
        return nullptr;
    }
}

char const* NVFP4PackedPlugin::getPluginName() const noexcept {
    return NVFP4_PACKED_PLUGIN_NAME;
}

char const* NVFP4PackedPlugin::getPluginVersion() const noexcept {
    return NVFP4_PACKED_PLUGIN_VERSION;
}

char const* NVFP4PackedPlugin::getPluginNamespace() const noexcept {
    return NVFP4_PACKED_PLUGIN_NAMESPACE;
}

int32_t NVFP4PackedPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
    // Store data type from activation input
    mDataType = in[0].desc.type;
    return 0;
}

int32_t NVFP4PackedPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
    // Output type matches activation input type
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t NVFP4PackedPlugin::getOutputShapes(
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

bool NVFP4PackedPlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    auto const& desc = inOut[pos].desc;

    // pos 0: activation [M, K] - FP32 or BF16
    // pos 1: weight_packed [N, K/2] - INT8
    // pos 2: scale_A [M, K/BLOCK_SIZE] - same as activation
    // pos 3: scale_W [N, K/BLOCK_SIZE] - same as activation
    // pos 4 (if hasBias): bias [N] - same as activation
    // last: output [M, N] - same as activation

    int outputPos = mHasBias ? 5 : 4;

    if (pos == 0) {
        // Activation: FP32 or BF16
        return (desc.type == nvinfer1::DataType::kFLOAT ||
                desc.type == nvinfer1::DataType::kBF16) &&
               desc.format == nvinfer1::TensorFormat::kLINEAR;
    } else if (pos == 1) {
        // Weight packed: INT8 (2 NVFP4 values per byte)
        return desc.type == nvinfer1::DataType::kINT8 &&
               desc.format == nvinfer1::TensorFormat::kLINEAR;
    } else if (pos == 2 || pos == 3) {
        // Scales: same type as activation
        return desc.type == inOut[0].desc.type &&
               desc.format == nvinfer1::TensorFormat::kLINEAR;
    } else if (mHasBias && pos == 4) {
        // Bias: same type as activation
        return desc.type == inOut[0].desc.type &&
               desc.format == nvinfer1::TensorFormat::kLINEAR;
    } else if (pos == outputPos) {
        // Output: same type as activation
        return desc.type == inOut[0].desc.type &&
               desc.format == nvinfer1::TensorFormat::kLINEAR;
    }
    return false;
}

size_t NVFP4PackedPlugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept {
    // No workspace needed for warp reduce version
    return 0;
}

int32_t NVFP4PackedPlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
    mBatchSize = in[0].dims.d[0];
    mDataType = in[0].type;
    return 0;
}

int32_t NVFP4PackedPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {

    int M = inputDesc[0].dims.d[0];  // Batch size
    int K = mInFeatures;
    int N = mOutFeatures;

    // Get pointers
    const void* activation = inputs[0];
    const void* weight_packed = inputs[1];
    const void* scale_A = inputs[2];
    const void* scale_W = inputs[3];
    const void* bias = mHasBias ? inputs[4] : nullptr;
    void* output = outputs[0];

    // Determine data type
    int data_type = 0;  // FP32
    if (mDataType == nvinfer1::DataType::kHALF) {
        data_type = 1;  // FP16
    } else if (mDataType == nvinfer1::DataType::kBF16) {
        data_type = 2;  // BF16
    }

    // Launch kernel
    nvfp4_gemv_forward(
        activation,
        weight_packed,
        scale_A,
        scale_W,
        bias,
        output,
        M, N, K,
        static_cast<int>(mActivationType),
        data_type,
        stream
    );

    return 0;
}

nvinfer1::IPluginV3* NVFP4PackedPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* context) noexcept {
    return clone();
}

nvinfer1::PluginFieldCollection const* NVFP4PackedPlugin::getFieldsToSerialize() noexcept {
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("in_features", &mInFeatures, nvinfer1::PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("out_features", &mOutFeatures, nvinfer1::PluginFieldType::kINT32, 1);

    int32_t actType = static_cast<int32_t>(mActivationType);
    mDataToSerialize.emplace_back("activation_type", &actType, nvinfer1::PluginFieldType::kINT32, 1);

    int32_t hasBias = mHasBias ? 1 : 0;
    mDataToSerialize.emplace_back("has_bias", &hasBias, nvinfer1::PluginFieldType::kINT32, 1);

    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ============================================================================
// NVFP4PackedPluginCreator Implementation
// ============================================================================

NVFP4PackedPluginCreator::NVFP4PackedPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "in_features", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "out_features", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "activation_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "has_bias", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

    mFC.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

char const* NVFP4PackedPluginCreator::getPluginName() const noexcept {
    return NVFP4_PACKED_PLUGIN_NAME;
}

char const* NVFP4PackedPluginCreator::getPluginVersion() const noexcept {
    return NVFP4_PACKED_PLUGIN_VERSION;
}

char const* NVFP4PackedPluginCreator::getPluginNamespace() const noexcept {
    return NVFP4_PACKED_PLUGIN_NAMESPACE;
}

nvinfer1::PluginFieldCollection const* NVFP4PackedPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

nvinfer1::IPluginV3* NVFP4PackedPluginCreator::createPlugin(
    char const* name,
    nvinfer1::PluginFieldCollection const* fc,
    nvinfer1::TensorRTPhase phase) noexcept {
    try {
        int inFeatures = 0;
        int outFeatures = 0;
        int activationType = 0;
        int hasBias = 0;

        for (int32_t i = 0; i < fc->nbFields; ++i) {
            auto const& field = fc->fields[i];
            if (std::strcmp(field.name, "in_features") == 0) {
                inFeatures = *static_cast<int32_t const*>(field.data);
            } else if (std::strcmp(field.name, "out_features") == 0) {
                outFeatures = *static_cast<int32_t const*>(field.data);
            } else if (std::strcmp(field.name, "activation_type") == 0) {
                activationType = *static_cast<int32_t const*>(field.data);
            } else if (std::strcmp(field.name, "has_bias") == 0) {
                hasBias = *static_cast<int32_t const*>(field.data);
            }
        }

        return new NVFP4PackedPlugin(
            inFeatures, outFeatures,
            static_cast<ActivationType>(activationType),
            hasBias != 0
        );
    } catch (...) {
        return nullptr;
    }
}

// Plugin registration
REGISTER_TENSORRT_PLUGIN(NVFP4PackedPluginCreator);

}  // namespace turbo_pi
