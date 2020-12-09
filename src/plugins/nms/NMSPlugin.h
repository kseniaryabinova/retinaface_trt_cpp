/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <NvInfer.h>

#include <string>
#include <vector>
#include <cassert>

#include "nms.h"

using namespace nvinfer1;

const std::string RETINAFACE_NMS_PLUGIN_NAME="RetinaNetNMS";
const std::string RETINAFACE_NMS_PLUGIN_VERSION="1";
const std::string RETINAFACE_NMS_PLUGIN_NAMESPACE="";

namespace retinaface {

    class NMSPlugin : public IPluginV2Ext {
        float _nms_thresh;
        int _detections_per_im;
        size_t _count;
        size_t _workspace_size = -1;


    protected:
        void deserialize(void const *data, size_t length);

        size_t getSerializationSize() const override;

        void serialize(void *buffer) const override;

    public:
        NMSPlugin(float nms_thresh, int detections_per_im);

        NMSPlugin(void const *data, size_t length);

        const char *getPluginType() const override {
            return RETINAFACE_NMS_PLUGIN_NAME.c_str();
        }

        const char *getPluginVersion() const override {
            return RETINAFACE_NMS_PLUGIN_VERSION.c_str();
        }

        int getNbOutputs() const override {
            return 3;
        }

        Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

        bool supportsFormat(DataType type, PluginFormat format) const override {
            return type == DataType::kFLOAT && format == PluginFormat::kNCHW;
        }

        int initialize() override { return 0; }

        void terminate() override {}

        size_t getWorkspaceSize(int maxBatchSize) const override;

        int enqueue(int batchSize,
                    const void *const *inputs, void **outputs,
                    void *workspace, cudaStream_t stream) override;

        void destroy() override {}

        const char *getPluginNamespace() const override {
            return RETINAFACE_NMS_PLUGIN_NAMESPACE.c_str();
        }

        void setPluginNamespace(const char *N) override { }

//        // IPluginV2Ext Methods
        DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const override {
            assert(index < 3);
            return DataType::kFLOAT;
        }

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted,
                                          int nbInputs) const override { return false; }

        bool canBroadcastInputAcrossBatch(int inputIndex) const override { return false; }

        void configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                             const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                             const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

        IPluginV2Ext *clone() const override {
            return new NMSPlugin(_nms_thresh, _detections_per_im);
        }

    private:
        template<typename T>
        void write(char *&buffer, const T &val) const {
            *reinterpret_cast<T *>(buffer) = val;
            buffer += sizeof(T);
        }

        template<typename T>
        void read(const char *&buffer, T &val) {
            val = *reinterpret_cast<const T *>(buffer);
            buffer += sizeof(T);
        }
    };

    class NMSPluginCreator : public IPluginCreator {
    public:
        NMSPluginCreator() {}

        const char *getPluginNamespace() const override {
            return RETINAFACE_NMS_PLUGIN_NAMESPACE.c_str();
        }

        const char *getPluginName() const override {
            return RETINAFACE_NMS_PLUGIN_NAME.c_str();
        }

        const char *getPluginVersion() const override {
            return RETINAFACE_NMS_PLUGIN_VERSION.c_str();
        }

        IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override {
            return new NMSPlugin(serialData, serialLength);
        }

        void setPluginNamespace(const char *N) override {}

        const PluginFieldCollection *getFieldNames() override { return nullptr; }

        IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override { return nullptr; }
    };

    REGISTER_TENSORRT_PLUGIN(NMSPluginCreator);

}

