
#include "DecodePlugin.h"


retinaface::DecodePlugin::DecodePlugin(void const *data, size_t length) {
    this->deserialize(data, length);
}

void retinaface::DecodePlugin::deserialize(void const *data, size_t length) {
    const char *d = static_cast<const char *>(data);
    read(d, _score_thresh);
    read(d, _top_n);
    read(d, _img_w);
    read(d, _img_h);
    read(d, _num_anchors);
    size_t anchors_size = _num_anchors;
    while (anchors_size--) {
        float val;
        read(d, val);
        _anchors.push_back(val);
    }
    read(d, _scale);
    read(d, _num_detections);
}

size_t retinaface::DecodePlugin::getSerializationSize() const {
    return sizeof(_score_thresh) + sizeof(_top_n)
        + sizeof(float) * _anchors.size() + sizeof(_scale)
        + sizeof(_num_anchors)
        + sizeof(_img_h) + sizeof(_img_w)
        + sizeof(_num_anchors) + sizeof(_num_detections); // + sizeof(_num_classes);
}

void retinaface::DecodePlugin::serialize(void *buffer) const {
    char *d = static_cast<char *>(buffer);
    write(d, _score_thresh);
    write(d, _top_n);
    write(d, _img_w);
    write(d, _img_h);
    write(d, _anchors.size());
    for (auto &val : _anchors) {
        write(d, val);
    }
    write(d, _scale);
    write(d, _num_detections);
}

retinaface::DecodePlugin::DecodePlugin(float score_thresh, int top_n, int scale, int num_detections,
        int img_h, int img_w, float* device_anchors)
        : _score_thresh(score_thresh), _top_n(top_n), _scale(scale), _num_detections(num_detections),
          _img_h(img_h), _img_w(img_w), _device_anchors(device_anchors) {

    _anchors = AnchorBoxes(false, img_w, img_h).get_anchors();
    _num_anchors = _anchors.size();
}

Dims retinaface::DecodePlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) {
    assert(nbInputDims == 3);
    assert(index < this->getNbOutputs());
    auto const &dims = inputs[index];
    return Dims3(_top_n * (index == 1 ? 1 : dims.d[1]), 1, 1);
}

int retinaface::DecodePlugin::initialize() {
    cudaMalloc(&_device_anchors, _anchors.size() * sizeof(float));
    cudaMemcpy(_device_anchors, _anchors.data(), _anchors.size() * sizeof(float), cudaMemcpyHostToDevice);

    return 0;
}

int retinaface::DecodePlugin::enqueue(int batchSize,
            const void *const *inputs, void **outputs,
            void *workspace, cudaStream_t stream) {
    if (_workspace_size == -1) {
        _workspace_size = getWorkspaceSize(batchSize);
    }

    return cuda::decode(batchSize, inputs, outputs,
            _img_h, _img_w, _device_anchors, _score_thresh, _top_n, _num_detections,
            workspace, _workspace_size, stream);
}

void retinaface::DecodePlugin::configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                     const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                     const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) {
    assert(*inputTypes == nvinfer1::DataType::kFLOAT &&
           floatFormat == nvinfer1::PluginFormat::kLINEAR);
    assert(nbInputs == 3);
    assert(nbOutputs == 3);
    auto const &boxes_dims = inputDims[0];
    auto const &scores_dims = inputDims[1];
    auto const &landmark_dims = inputDims[2];
    assert(boxes_dims.d[1] == 4);
    assert(scores_dims.d[1] == 2);
    assert(landmark_dims.d[1] == 10);
}
