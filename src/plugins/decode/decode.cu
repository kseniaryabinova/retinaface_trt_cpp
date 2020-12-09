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

#include "decode.h"
#include "../utils.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/tabulate.h>
#include <thrust/count.h>
#include <thrust/find.h>
#include <thrust/extrema.h>

#include <cub/cub.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

namespace retinaface {
namespace cuda {

inline __device__ float clamp(const float& val, const float& min_val, const float& max_val) {
    return min(max_val, max(val, min_val));
}


int decode(int batch_size,
      const void *const *inputs, void **outputs,
      size_t height, size_t width, float* device_anchors,
      float score_thresh, int top_n, size_t num_detections,
      void *workspace, size_t workspace_size, cudaStream_t stream) {

    int scores_size = num_detections;

    if (!workspace_size) {
      // Return required scratch space size cub style
        workspace_size = get_size_aligned<bool>(scores_size);     // flags
        workspace_size += get_size_aligned<int>(scores_size);      // indices
        workspace_size += get_size_aligned<int>(scores_size);      // indices_sorted
        workspace_size += get_size_aligned<float>(scores_size);    // scores
        workspace_size += get_size_aligned<float>(scores_size);    // scores
        workspace_size += get_size_aligned<float>(scores_size);    // scores_sorted

        size_t temp_size_flag = 0;
        cub::DeviceSelect::Flagged((void *)nullptr, temp_size_flag,
          cub::CountingInputIterator<int>(scores_size),
          (bool *)nullptr, (int *)nullptr, (int *)nullptr, scores_size);
        size_t temp_size_sort = 0;
        cub::DeviceRadixSort::SortPairsDescending((void *)nullptr, temp_size_sort,
          (float *)nullptr, (float *)nullptr, (int *)nullptr, (int *)nullptr, scores_size);
        workspace_size += std::max(temp_size_flag, temp_size_sort);

        return workspace_size;
    }

    auto on_stream = thrust::cuda::par.on(stream);

    auto flags = get_next_ptr<bool>(scores_size, workspace, workspace_size);
    auto indices = get_next_ptr<int>(scores_size, workspace, workspace_size);
    auto indices_sorted = get_next_ptr<int>(scores_size, workspace, workspace_size);
    auto scores1 = get_next_ptr<float>(scores_size, workspace, workspace_size);
    auto scores2 = get_next_ptr<float>(scores_size, workspace, workspace_size);
    auto scores_sorted = get_next_ptr<float>(scores_size, workspace, workspace_size);

    for (int batch = 0; batch < batch_size; ++batch) {
        auto in_boxes = static_cast<const float *>(inputs[0]) + batch * scores_size * 4;
        auto in_scores = static_cast<const float *>(inputs[1]) + batch * scores_size * 2;
        auto in_landmarks = static_cast<const float*>(inputs[2]) + batch * scores_size * 10;

        auto out_boxes = static_cast<float4 *>(outputs[0]) + batch * top_n;
        auto out_scores = static_cast<float *>(outputs[1]) + batch * top_n;
        auto out_landmarks = static_cast<float *>(outputs[2]) + batch * top_n * 10;

        // scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        thrust::transform(on_stream,
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(scores_size),
                scores1,
                [=] __device__ (const int& index){
                    return in_scores[index*2+1];
        });

        // Discard scores below threshold
        thrust::transform(on_stream, scores1, scores1 + scores_size,
                          flags, thrust::placeholders::_1 > score_thresh);

        int *num_selected = reinterpret_cast<int *>(indices_sorted);
        cub::DeviceSelect::Flagged(workspace, workspace_size,
          cub::CountingInputIterator<int>(0),
          flags, indices, num_selected, scores_size, stream);
        cudaStreamSynchronize(stream);
        int num_detections = *thrust::device_pointer_cast(num_selected);

        // Only keep top n scores
        auto indices_filtered = indices;
        if (num_detections > top_n) {
            thrust::gather(on_stream, indices, indices + num_detections,
                           scores1, scores2);
            cub::DeviceRadixSort::SortPairsDescending(
                    workspace, workspace_size,
                    scores2, scores_sorted, indices, indices_sorted,
                    num_detections, 0, sizeof(*scores2) * 8, stream);
            indices_filtered = indices_sorted;
            num_detections = top_n;
        }

        // по четным
        for (int j=0; j<10; j += 2){
            thrust::transform(on_stream,
                              indices_filtered, indices_filtered + num_detections,
                              thrust::make_permutation_iterator(out_landmarks+j, thrust::make_transform_iterator(
                                      thrust::make_counting_iterator(0), [=] __device__ (const int& i) { return i * 10; })),
            [=] __device__ (const int& i) {
                return clamp((device_anchors[i*4+0] + in_landmarks[i*10+j]*0.1*device_anchors[i*4+2]) * width, 0, width);
            });
        }

        // по нечетным
        for (int j=1; j<10; j += 2){
            thrust::transform(on_stream,
                              indices_filtered, indices_filtered + num_detections,
                              thrust::make_permutation_iterator(out_landmarks+j, thrust::make_transform_iterator(
                                      thrust::make_counting_iterator(0), [=] __device__ (const int& i) { return i* 10; })),
            [=] __device__ (const int& i) {
                return clamp((device_anchors[i*4+1] + in_landmarks[i*10+j]*0.1*device_anchors[i*4+3]) * height, 0, height);
            });
        }

        thrust::transform(on_stream,
                indices_filtered, indices_filtered + num_detections,
            thrust::make_zip_iterator(thrust::make_tuple(out_scores, out_boxes)),
            [=] __device__ (const int& i) {

                float x1 = device_anchors[i*4] + in_boxes[i*4] * 0.1 * device_anchors[i*4+2];
                float y1 = device_anchors[i*4+1] + in_boxes[i*4+1] * 0.1 * device_anchors[i*4+3];
                float x2 = device_anchors[i*4+2] * __expf(in_boxes[i*4+2] * 0.2);
                float y2 = device_anchors[i*4+3] * __expf(in_boxes[i*4+3] * 0.2);
                x1 -= x2 / 2.;
                y1 -= y2 / 2.;
                x2 += x1;
                y2 += y1;

                return thrust::make_tuple(
                        in_scores[i*2+1],
                        float4{
                            clamp(x1*width, 0, width),
                            clamp(y1*height, 0, height),
                            clamp(x2*width, 0, width),
                            clamp(y2*height, 0, height)
                        });
        });

        // Zero-out unused scores
        if (num_detections < top_n) {
          thrust::fill(on_stream, out_scores + num_detections,
            out_scores + top_n, 0.0f);
        }
    }

    return 0;
}

}
}
