
#include "NMSPlugin.h"


retinaface::NMSPlugin::NMSPlugin(void const *data, size_t length) {
    this->deserialize(data, length);
}

retinaface::NMSPlugin::NMSPlugin(float nms_thresh, int detections_per_im)
        : _nms_thresh(nms_thresh), _detections_per_im(detections_per_im) {
    assert(nms_thresh > 0);
    assert(detections_per_im > 0);
}


void retinaface::NMSPlugin::deserialize(void const *data, size_t length) {
    const char *d = static_cast<const char *>(data);
    read(d, _nms_thresh);
    read(d, _detections_per_im);
    read(d, _count);
}

size_t retinaface::NMSPlugin::getSerializationSize() const {
    return sizeof(_nms_thresh) + sizeof(_detections_per_im)
                   + sizeof(_count);
}

void retinaface::NMSPlugin::serialize(void *buffer) const {
    char *d = static_cast<char *>(buffer);
    write(d, _nms_thresh);
    write(d, _detections_per_im);
    write(d, _count);
}

Dims retinaface::NMSPlugin::getOutputDimensions(int index,
                         const Dims *inputs, int nbInputDims)  {
    assert(nbInputDims == 3);
    assert(index < this->getNbOutputs());
    auto const &dims = inputs[index];
    return Dims3(dims.d[0], 1, 1);
}

size_t retinaface::NMSPlugin::getWorkspaceSize(int maxBatchSize) const  {
    int size = cuda::nms(maxBatchSize, nullptr, nullptr, _count,
                         _detections_per_im, _nms_thresh,
                         nullptr, 0, nullptr);
    return size;
}

int retinaface::NMSPlugin::enqueue(int batchSize,
                const void *const *inputs, void **outputs,
                void *workspace, cudaStream_t stream)  {

    if (_workspace_size == -1)
    _workspace_size = getWorkspaceSize(batchSize);

    return cuda::nms(batchSize, inputs, outputs, _count,
            _detections_per_im, _nms_thresh,
            workspace, _workspace_size, stream);
}

void retinaface::NMSPlugin::configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                         const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                         const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)  {
    assert(*inputTypes == nvinfer1::DataType::kFLOAT &&
           floatFormat == nvinfer1::PluginFormat::kLINEAR);
    assert(nbInputs == 3);
    assert(nbOutputs == 3);
}

