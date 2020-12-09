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

#include <cuda_runtime.h>

#include <NvInfer.h>

#include <string>
#include <cassert>
#include <vector>
#include <cmath>

#include "decode.h"

using namespace nvinfer1;

const std::string RETINAFACE_DECODE_PLUGIN_NAME="DecodePlugin";
const std::string RETINAFACE_DECODE_PLUGIN_VERSION="1";
const std::string RETINAFACE_DECODE_PLUGIN_NAMESPACE="";

namespace retinaface {

    class DecodePlugin : public IPluginV2Ext {
        class AnchorBoxes {
            float min_sizes[3][2];
            float steps[3];
            int feature_maps[3][2];

            bool clip;
            int image_size[2];

        public:
            AnchorBoxes(bool clip, int width, int height) :
                    min_sizes{{16,  32},
                              {64,  128},
                              {256, 512}},
                    steps{8, 16, 32},
                    image_size{height, width} {
                this->clip = clip;

                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        this->feature_maps[i][j] = ceil(this->image_size[j] / this->steps[i]);
                    }
                }
            }

            std::vector<float> get_anchors() {
                std::vector<float> anchors;
                int counter = 0;

                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < this->feature_maps[i][0]; ++j) {
                        for (int k = 0; k < this->feature_maps[i][1]; ++k) {
                            for (int t = 0; t < 2; ++t) {
                                auto s_kx = this->min_sizes[i][t] / this->image_size[1];
                                auto s_ky = this->min_sizes[i][t] / this->image_size[0];
                                auto dense_cx = (float(k) + 0.5) * this->steps[i] / this->image_size[1];
                                auto dense_cy = (float(j) + 0.5) * this->steps[i] / this->image_size[0];

                                anchors.push_back(dense_cx);
                                anchors.push_back(dense_cy);
                                anchors.push_back(s_kx);
                                anchors.push_back(s_ky);
                                counter += 4;
                            }
                        }
                    }
                }

                return anchors;
            }
        };

        float _score_thresh;
        int _top_n;
        std::vector<float> _anchors;
        float *_device_anchors;
        float _scale;
        int _img_w;
        int _img_h;

        size_t _num_anchors;
        size_t _num_detections;

        size_t _workspace_size = -1;

    protected:
        void deserialize(void const *data, size_t length);

        size_t getSerializationSize() const override;

        void serialize(void *buffer) const override;

    public:
        DecodePlugin(float score_thresh, int top_n, int scale, int num_detections,
                int img_h, int img_w, float* device_anchors=NULL);

        DecodePlugin(void const *data, size_t length);

        const char *getPluginType() const override {
            return RETINAFACE_DECODE_PLUGIN_NAME.c_str();
        }

        const char *getPluginVersion() const override {
            return RETINAFACE_DECODE_PLUGIN_VERSION.c_str();
        }

        int getNbOutputs() const override {
            return 3;
        }

        Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

        bool supportsFormat(DataType type, PluginFormat format) const override {
            return type == DataType::kFLOAT && format == PluginFormat::kNCHW;
        }

        int initialize() override;

        void terminate() override {}

        size_t getWorkspaceSize(int maxBatchSize) const override {
            int size = cuda::decode(maxBatchSize, nullptr, nullptr,
                                    _img_h, _img_w, _device_anchors, _score_thresh, _top_n, _num_detections,
                                    nullptr, 0, nullptr);
            return size;
        }

        int enqueue(int batchSize,
                    const void *const *inputs, void **outputs,
                    void *workspace, cudaStream_t stream) override;

        void destroy() override {};

        const char *getPluginNamespace() const override {
            return RETINAFACE_DECODE_PLUGIN_NAMESPACE.c_str();
        }

        void setPluginNamespace(const char *N) override {}

        // IPluginV2Ext Methods
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
            return new DecodePlugin(_score_thresh, _top_n, _scale, _num_detections, _img_h, _img_w, _device_anchors);
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

    class DecodePluginCreator : public IPluginCreator {
    public:
        DecodePluginCreator() {}

        const char *getPluginName() const override {
            return RETINAFACE_DECODE_PLUGIN_NAME.c_str();
        }

        const char *getPluginVersion() const override {
            return RETINAFACE_DECODE_PLUGIN_VERSION.c_str();
        }

        const char *getPluginNamespace() const override {
            return RETINAFACE_DECODE_PLUGIN_NAMESPACE.c_str();
        }

        IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override {
            return new DecodePlugin(serialData, serialLength);
        }

        void setPluginNamespace(const char *N) override {}

        const PluginFieldCollection *getFieldNames() override { return nullptr; }

        IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override { return nullptr; }
    };

    REGISTER_TENSORRT_PLUGIN(DecodePluginCreator);

}

