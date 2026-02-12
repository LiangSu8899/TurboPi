/**
 * W4A16 MLP TensorRT Plugin Implementation
 *
 * Implements the IPluginV3 interface for W4A16 quantized MLP layers.
 *
 * Author: Claude Code
 * Date: 2026-02-10
 */

#include "w4a16_mlp_plugin.h"
#include <cassert>
#include <cstring>

namespace turbo_pi {

// ============================================================================
// W4A16MLPPlugin Implementation
// ============================================================================

W4A16MLPPlugin::W4A16MLPPlugin(int inFeatures, int outFeatures, LayerType layerType, int blockSize)
    : mInFeatures(inFeatures)
    , mOutFeatures(outFeatures)
    , mBlockSize(blockSize)
    , mBatchSize(1)
    , mLayerType(layerType)
{
}

W4A16MLPPlugin::W4A16MLPPlugin(const W4A16MLPPlugin& other)
    : mInFeatures(other.mInFeatures)
    , mOutFeatures(other.mOutFeatures)
    , mBlockSize(other.mBlockSize)
    , mBatchSize(other.mBatchSize)
    , mLayerType(other.mLayerType)
{
}

nvinfer1::IPluginCapability* W4A16MLPPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept
{
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

nvinfer1::IPluginV3* W4A16MLPPlugin::clone() noexcept
{
    try {
        auto* plugin = new W4A16MLPPlugin(*this);
        return plugin;
    } catch (...) {
        return nullptr;
    }
}

int32_t W4A16MLPPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    // Inputs:
    // 0: A [M, K] float32 activation
    // 1: W_packed [N, K/2] uint8 packed FP4 weights
    // 2: scales [N, num_blocks_k] float32 weight scales
    // Output:
    // 0: C [M, N] float32

    if (nbInputs != 3 || nbOutputs != 1) {
        return -1;
    }

    // Extract dimensions from input shapes
    auto const& aDesc = in[0].desc;
    mBatchSize = aDesc.dims.d[0];  // M
    mInFeatures = aDesc.dims.d[1]; // K

    auto const& wDesc = in[1].desc;
    mOutFeatures = wDesc.dims.d[0]; // N

    return 0;
}

int32_t W4A16MLPPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (nbOutputs != 1) {
        return -1;
    }
    // Output is float32
    outputTypes[0] = nvinfer1::DataType::kFLOAT;
    return 0;
}

int32_t W4A16MLPPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (nbInputs != 3 || nbOutputs != 1) {
        return -1;
    }

    // A: [M, K], W_packed: [N, K/2] -> C: [M, N]
    outputs[0].nbDims = 2;
    outputs[0].d[0] = inputs[0].d[0]; // M (batch)
    outputs[0].d[1] = inputs[1].d[0]; // N (out features)

    return 0;
}

bool W4A16MLPPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // pos 0: A (float32, linear)
    // pos 1: W_packed (int8/uint8, linear)
    // pos 2: scales (float32, linear)
    // pos 3: C output (float32, linear)

    if (pos == 0) {
        // Activation: float32
        return inOut[pos].desc.type == nvinfer1::DataType::kFLOAT &&
               inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;
    } else if (pos == 1) {
        // Packed weights: int8 (TRT doesn't have uint8, we treat as int8)
        return inOut[pos].desc.type == nvinfer1::DataType::kINT8 &&
               inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;
    } else if (pos == 2) {
        // Scales: float32
        return inOut[pos].desc.type == nvinfer1::DataType::kFLOAT &&
               inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;
    } else if (pos == 3) {
        // Output: float32
        return inOut[pos].desc.type == nvinfer1::DataType::kFLOAT &&
               inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;
    }
    return false;
}

size_t W4A16MLPPlugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    // No additional workspace needed
    return 0;
}

int32_t W4A16MLPPlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    // Update dimensions based on actual shapes
    mBatchSize = in[0].dims.d[0];
    mInFeatures = in[0].dims.d[1];
    mOutFeatures = in[1].dims.d[0];
    return 0;
}

int32_t W4A16MLPPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    // Get input pointers
    const float* A = static_cast<const float*>(inputs[0]);
    const uint8_t* W_packed = static_cast<const uint8_t*>(inputs[1]);
    const float* scales = static_cast<const float*>(inputs[2]);
    float* C = static_cast<float*>(outputs[0]);

    // Get dimensions
    int M = inputDesc[0].dims.d[0];
    int K = inputDesc[0].dims.d[1];
    int N = inputDesc[1].dims.d[0];

    // Launch appropriate kernel based on layer type and dimensions
    if (M == 1) {
        // Single token inference - use optimized GEMV
        if (mLayerType == LayerType::GATE_PROJ || mLayerType == LayerType::UP_PROJ) {
            // gate_proj/up_proj: K=2048, N=16384
            launch_w4a16_gate_up_proj(A, W_packed, scales, C, stream);
        } else if (mLayerType == LayerType::DOWN_PROJ) {
            // down_proj: K=16384, N=2048
            launch_w4a16_down_proj(A, W_packed, scales, C, stream);
        } else {
            // Generic dimensions
            launch_w4a16_gemv(A, W_packed, scales, C, N, K, stream);
        }
    } else {
        // Batched inference - process row by row
        for (int m = 0; m < M; ++m) {
            launch_w4a16_gemv(
                A + m * K,
                W_packed,
                scales,
                C + m * N,
                N, K, stream
            );
        }
    }

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

    mDataToSerialize.emplace_back(nvinfer1::PluginField{
        "in_features", &mInFeatures, nvinfer1::PluginFieldType::kINT32, 1});
    mDataToSerialize.emplace_back(nvinfer1::PluginField{
        "out_features", &mOutFeatures, nvinfer1::PluginFieldType::kINT32, 1});
    mDataToSerialize.emplace_back(nvinfer1::PluginField{
        "block_size", &mBlockSize, nvinfer1::PluginFieldType::kINT32, 1});
    int layerTypeInt = static_cast<int>(mLayerType);
    mDataToSerialize.emplace_back(nvinfer1::PluginField{
        "layer_type", &layerTypeInt, nvinfer1::PluginFieldType::kINT32, 1});

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();

    return &mFCToSerialize;
}

// ============================================================================
// W4A16MLPPluginCreator Implementation
// ============================================================================

W4A16MLPPluginCreator::W4A16MLPPluginCreator()
{
    mPluginAttributes.emplace_back(nvinfer1::PluginField{
        "in_features", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    mPluginAttributes.emplace_back(nvinfer1::PluginField{
        "out_features", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    mPluginAttributes.emplace_back(nvinfer1::PluginField{
        "block_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    mPluginAttributes.emplace_back(nvinfer1::PluginField{
        "layer_type", nullptr, nvinfer1::PluginFieldType::kINT32, 1});

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
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
    try {
        int inFeatures = 0;
        int outFeatures = 0;
        int blockSize = 32;
        int layerType = 3;  // GENERIC

        for (int i = 0; i < fc->nbFields; ++i) {
            const auto& field = fc->fields[i];
            if (strcmp(field.name, "in_features") == 0) {
                inFeatures = *static_cast<const int*>(field.data);
            } else if (strcmp(field.name, "out_features") == 0) {
                outFeatures = *static_cast<const int*>(field.data);
            } else if (strcmp(field.name, "block_size") == 0) {
                blockSize = *static_cast<const int*>(field.data);
            } else if (strcmp(field.name, "layer_type") == 0) {
                layerType = *static_cast<const int*>(field.data);
            }
        }

        return new W4A16MLPPlugin(
            inFeatures, outFeatures,
            static_cast<W4A16MLPPlugin::LayerType>(layerType),
            blockSize
        );
    } catch (...) {
        return nullptr;
    }
}

}  // namespace turbo_pi
