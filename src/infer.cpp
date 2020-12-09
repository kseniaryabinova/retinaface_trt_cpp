#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "engine.h"

using namespace std;
using namespace cv;


cv::Mat prepare_image(const cv::Mat& image, const cv::Size& net_size) {
    cv::Mat temp_image;
    int max_border = std::max(image.size().width, image.size().height);
    int y_offset = (max_border - image.size().height) / 2;
    int x_offset = (max_border - image.size().width) / 2;

    cv::copyMakeBorder(image, temp_image, y_offset, y_offset, x_offset, x_offset, cv::BORDER_CONSTANT,
                       cv::Scalar(127.5, 127.5, 127.5));

    cv::resize(temp_image, temp_image, net_size, 0, 0, cv::INTER_CUBIC);
    cv::cvtColor(temp_image, temp_image, CV_BGR2RGB);

    return temp_image;
}


int main(int argc, char *argv[]) {
	if (argc != 3) {
		cerr << "Usage: " << argv[0] << " engine.plan image.jpg" << endl;
		return 1;
	}

	cout << "Loading engine..." << endl;
    cudaProfilerStart();
	auto engine = retinaface::Engine(argv[1]);
    auto input_dims = engine.getInputSize();

	cout << "Preparing data..." << endl;
	auto image = imread(argv[2], IMREAD_COLOR);
//	image = prepare_image(image, cv::Size(1024, 1024));
    cv::resize(image, image, Size(input_dims[1], input_dims[0]), 0, 0);
    cv::cvtColor(image, image, CV_BGR2RGB);

    cv::Mat pixels;
    image.convertTo(pixels, CV_32FC3);

    int channels = 3;
    vector<float> img;
    vector<float> data (channels * input_dims[0] * input_dims[1]);

    if (pixels.isContinuous())
        img.assign((float*)pixels.datastart, (float*)pixels.dataend);
    else {
        cerr << "Error reading image " << argv[2] << endl;
        return -1;
    }

    vector<float> mean {104, 117, 123};

    for (int c = 0; c < channels; c++) {
        for (int j = 0, hw = input_dims[0] * input_dims[1]; j < hw; j++) {
            data[c * hw + j] = img[channels * j + 2 - c] - mean[c];
        }
    }

    void *data_d, *scores_d, *boxes_d, *landmarks_d;
    auto num_det = engine.getMaxDetections();
    cudaMalloc(&data_d, 3 * input_dims[0] * input_dims[1] * sizeof(float));

    // Copy image to device
    cudaMemcpy(data_d, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

	// Create device buffers
	cudaMalloc(&scores_d, num_det * sizeof(float));
	cudaMalloc(&boxes_d, num_det * 4 * sizeof(float));
	cudaMalloc(&landmarks_d, num_det * 10 * sizeof(float));

    // Get back the bounding boxes
    auto scores = new float[num_det];
    auto boxes = new float[num_det * 4];
    auto landmarks = new float[num_det * 10];

	// Run inference n times
	cout << "Running inference..." << endl;
 	vector<void *> buffers = { data_d, boxes_d, scores_d, landmarks_d };
    for (int i=0; i<10; ++i){
        auto start = chrono::system_clock::now();
        engine.infer(buffers, 1);
        auto stop = chrono::system_clock::now();
        auto timing = chrono::duration_cast<chrono::duration<double>>(stop - start);
        cout << "Took " << timing.count() << " seconds per inference." << endl;
    }

    cudaMemcpy(scores, scores_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);
    cudaMemcpy(boxes, boxes_d, sizeof(float) * num_det * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(landmarks, landmarks_d, sizeof(float) * num_det * 10, cudaMemcpyDeviceToHost);

    cudaProfilerStop();
    int count = 0;

	for (int i = 0; i < num_det; i++) {
		// Show results over confidence threshold
		if (scores[i] >= 0.6f) {
			int x1 = int(boxes[i*4+0]);
			int y1 = int(boxes[i*4+1]);
			int x2 = int(boxes[i*4+2]);
			int y2 = int(boxes[i*4+3]);

            cv::rectangle(image, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 0), 1);

//            cout << "Found box index=" << i
//            << " score=[" << scores[i] //<< ", " << scores[i*2+1]
//            << "]  {" << boxes[i*4] << ", " << boxes[i*4+1]
//            << ", " << boxes[i*4+2] << ", " << boxes[i*4+3] << "}  {"
//            << landmarks[0+10*i] << ", " << landmarks[1+10*i] << ", "
//            << landmarks[2+10*i] << ", " << landmarks[3+10*i] << ", "
//            << landmarks[4+10*i] << ", " << landmarks[5+10*i] << ", "
//            << landmarks[6+10*i] << ", " << landmarks[7+10*i] << ", "
//            << landmarks[8+10*i] << ", " << landmarks[9+10*i]
//            << "}" << endl;

            auto p1 = Point(int(landmarks[0+10*i]), int(landmarks[1+10*i]));
            auto p2 = Point(int(landmarks[2+10*i]), int(landmarks[3+10*i]));
            auto p3 = Point(int(landmarks[4+10*i]), int(landmarks[5+10*i]));
            auto p4 = Point(int(landmarks[6+10*i]), int(landmarks[7+10*i]));
            auto p5 = Point(int(landmarks[8+10*i]), int(landmarks[9+10*i]));

			cv::circle(image, p1, 1, cv::Scalar(0, 0, 255), 1);
			cv::circle(image, p2, 1, cv::Scalar(0, 255, 255), 1);
			cv::circle(image, p3, 1, cv::Scalar(255, 0, 255), 1);
			cv::circle(image, p4, 1, cv::Scalar(0, 255, 0), 1);
			cv::circle(image, p5, 1, cv::Scalar(255, 0, 0), 1);

			++count;
		}
	}

	printf("count=%d\n", count);

	delete[] scores;
	delete[] boxes;
	delete[] landmarks;

	// Write image
	cv::cvtColor(image, image, CV_RGB2BGR);
	imwrite("result.jpg", image);

	return 0;
}
