/*
 * W4A16 MLP TensorRT Plugin Implementation
 */

#include "w4a16_trt_plugin.h"
#include "w4a16_tvm_kernels.h"
#include <cstring>
#include <iostream>
#include <cassert>

namespace turbo_pi {

// Static plugin attributes
nvinfer1::PluginFieldCollection W4A16MLPPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> W4A16MLPPluginCreator::mPluginAttributes{};

// =============================================================================
// W4A16MLPPlugin Implementation
// =============================================================================

W4A16MLPPlugin::W4A16MLPPlugin(int hiddenSize, int intermediateSize, const std::string& libDir)
    : mHiddenSize(hiddenSize)
    , mIntermediateSize(intermediateSize)
    , mLibDir(libDir)
    , mKernels(nullptr)
{
}

W4A16MLPPlugin::W4A16MLPPlugin(const W4A16MLPPlugin& other)
    : mHiddenSize(other.mHiddenSize)
    , mIntermediateSize(other.mIntermediateSize)
    , mLibDir(other.mLibDir)
    , mKernels(nullptr)  // Don't copy, will be re-created
{
}

W4A16MLPPlugin::~W4A16MLPPlugin() = default;

nvinfer1::IPluginCapability* W4A16MLPPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept
{
    if (type == nvinfer1::PluginCapabilityType::kCORE) {
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kBUILD) {
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kRUNTIME) {
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

nvinfer1::IPluginV3* W4A16MLPPlugin::clone() noexcept
{
    try {
        auto* plugin = new W4A16MLPPlugin(*this);
        return plugin;
    } catch (const std::exception& e) {
        std::cerr << "W4A16MLPPlugin clone failed: " << e.what() << std::endl;
        return nullptr;
    }
}

char const* W4A16MLPPlugin::getPluginName() const noexcept
{
    return W4A16_MLP_PLUGIN_NAME;
}

char const* W4A16MLPPlugin::getPluginVersion() const noexcept
{
    return W4A16_MLP_PLUGIN_VERSION;
}

char const* W4A16MLPPlugin::getPluginNamespace() const noexcept
{
    return W4A16_MLP_PLUGIN_NAMESPACE;
}

int32_t W4A16MLPPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out,
    int32_t nbOutputs) noexcept
{
    // Validate input/output counts
    // Inputs: x, gate_W, gate_S, up_W, up_S, down_W, down_S (7 inputs)
    // Outputs: output (1 output)
    assert(nbInputs == 7);
    assert(nbOutputs == 1);

    return 0;
}

int32_t W4A16MLPPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept
{
    assert(nbOutputs == 1);
    outputTypes[0] = nvinfer1::DataType::kFLOAT;  // FP32 output
    return 0;
}

int32_t W4A16MLPPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::DimsExprs const* shapeInputs,
    int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Output shape is same as input x: [1, hidden_size]
    outputs[0].nbDims = 2;
    outputs[0].d[0] = exprBuilder.constant(1);
    outputs[0].d[1] = exprBuilder.constant(mHiddenSize);
    return 0;
}

bool W4A16MLPPlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept
{
    // All tensors must be on GPU and in linear format
    const auto& desc = inOut[pos];
    if (desc.desc.format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }

    // Input tensors:
    // 0: x [1, H] - FP32
    // 1: gate_W_packed [I, H/2] - INT8 (uint8)
    // 2: gate_scales [I, num_blocks_h] - FP32
    // 3: up_W_packed [I, H/2] - INT8 (uint8)
    // 4: up_scales [I, num_blocks_h] - FP32
    // 5: down_W_packed [H, I/2] - INT8 (uint8)
    // 6: down_scales [H, num_blocks_i] - FP32
    // 7: output [1, H] - FP32

    if (pos == 0) {  // x
        return desc.desc.type == nvinfer1::DataType::kFLOAT;
    } else if (pos == 1 || pos == 3 || pos == 5) {  // packed weights
        return desc.desc.type == nvinfer1::DataType::kINT8;
    } else if (pos == 2 || pos == 4 || pos == 6) {  // scales
        return desc.desc.type == nvinfer1::DataType::kFLOAT;
    } else if (pos == 7) {  // output
        return desc.desc.type == nvinfer1::DataType::kFLOAT;
    }

    return false;
}

size_t W4A16MLPPlugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept
{
    // Workspace for intermediate results:
    // gate_out [1, I] + up_out [1, I] = 2 * I floats
    return 2 * mIntermediateSize * sizeof(float);
}

int32_t W4A16MLPPlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out,
    int32_t nbOutputs) noexcept
{
    // Shape is fixed for this plugin
    return 0;
}

int32_t W4A16MLPPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept
{
    // Lazy initialize TVM kernels
    if (!mKernels) {
        try {
            mKernels = std::make_unique<W4A16TVMKernels>(mLibDir);
        } catch (const std::exception& e) {
            std::cerr << "W4A16MLPPlugin: Failed to load TVM kernels: " << e.what() << std::endl;
            return -1;
        }
    }

    // Extract input pointers
    const float* x = static_cast<const float*>(inputs[0]);
    const uint8_t* gate_W = static_cast<const uint8_t*>(inputs[1]);
    const float* gate_S = static_cast<const float*>(inputs[2]);
    const uint8_t* up_W = static_cast<const uint8_t*>(inputs[3]);
    const float* up_S = static_cast<const float*>(inputs[4]);
    const uint8_t* down_W = static_cast<const uint8_t*>(inputs[5]);
    const float* down_S = static_cast<const float*>(inputs[6]);

    // Output pointer
    float* out = static_cast<float*>(outputs[0]);

    // Workspace for intermediate results
    float* work = static_cast<float*>(workspace);

    // Execute full MLP using TVM kernels
    mKernels->fullMLP(
        x,
        gate_W, gate_S,
        up_W, up_S,
        down_W, down_S,
        out,
        work,
        stream
    );

    return 0;
}

nvinfer1::IPluginV3* W4A16MLPPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* context) noexcept
{
    return clone();
}

nvinfer1::PluginFieldCollection const* W4A16MLPPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();

    mDataToSerialize.push_back(nvinfer1::PluginField(
        "hidden_size", &mHiddenSize, nvinfer1::PluginFieldType::kINT32, 1));
    mDataToSerialize.push_back(nvinfer1::PluginField(
        "intermediate_size", &mIntermediateSize, nvinfer1::PluginFieldType::kINT32, 1));
    mDataToSerialize.push_back(nvinfer1::PluginField(
        "lib_dir", mLibDir.c_str(), nvinfer1::PluginFieldType::kCHAR, mLibDir.length()));

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();

    return &mFCToSerialize;
}

// =============================================================================
// W4A16MLPPluginCreator Implementation
// =============================================================================

W4A16MLPPluginCreator::W4A16MLPPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.push_back(nvinfer1::PluginField(
        "hidden_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.push_back(nvinfer1::PluginField(
        "intermediate_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.push_back(nvinfer1::PluginField(
        "lib_dir", nullptr, nvinfer1::PluginFieldType::kCHAR, 0));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* W4A16MLPPluginCreator::getPluginName() const noexcept
{
    return W4A16_MLP_PLUGIN_NAME;
}

char const* W4A16MLPPluginCreator::getPluginVersion() const noexcept
{
    return W4A16_MLP_PLUGIN_VERSION;
}

char const* W4A16MLPPluginCreator::getPluginNamespace() const noexcept
{
    return W4A16_MLP_PLUGIN_NAMESPACE;
}

nvinfer1::PluginFieldCollection const* W4A16MLPPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV3* W4A16MLPPluginCreator::createPlugin(
    char const* name,
    nvinfer1::PluginFieldCollection const* fc,
    nvinfer1::TensorRTPhase phase) noexcept
{
    int hiddenSize = DEFAULT_HIDDEN_SIZE;
    int intermediateSize = DEFAULT_INTERMEDIATE_SIZE;
    std::string libDir = "";

    for (int i = 0; i < fc->nbFields; ++i) {
        const auto& field = fc->fields[i];
        if (strcmp(field.name, "hidden_size") == 0) {
            hiddenSize = *static_cast<const int*>(field.data);
        } else if (strcmp(field.name, "intermediate_size") == 0) {
            intermediateSize = *static_cast<const int*>(field.data);
        } else if (strcmp(field.name, "lib_dir") == 0) {
            libDir = std::string(static_cast<const char*>(field.data), field.length);
        }
    }

    try {
        return new W4A16MLPPlugin(hiddenSize, intermediateSize, libDir);
    } catch (const std::exception& e) {
        std::cerr << "W4A16MLPPluginCreator: Failed to create plugin: " << e.what() << std::endl;
        return nullptr;
    }
}

// =============================================================================
// Plugin Registration
// =============================================================================

REGISTER_TENSORRT_PLUGIN(W4A16MLPPluginCreator);

extern "C" bool initLibW4A16MLPPlugin()
{
    return true;
}

}  // namespace turbo_pi
