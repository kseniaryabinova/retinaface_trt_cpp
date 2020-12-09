#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <experimental/filesystem>

#include "engine.h"

using namespace std;
namespace fs = std::experimental::filesystem;

// Sample program to build a TensorRT Engine from an ONNX model from RetinaNet
//
// By default TensorRT will target FP16 precision (supported on Pascal, Volta, and Turing GPUs)
//
// You can optionally provide an INT8CalibrationTable file created during RetinaNet INT8 calibration
// to build a TensorRT engine with INT8 precision

int main(int argc, char *argv[]) {
	if (argc != 3 && argc != 4) {
		cerr << "Usage: " << argv[0] << " model.onnx model.plan {Int8CalibrationTable}" << endl;
		return 1;
	}

	ifstream onnxFile;
	onnxFile.open(argv[1], ios::in | ios::binary); 

        if (!onnxFile.good()) {
            cerr << "\nERROR: Unable to read specified ONNX model " << argv[1] << endl;
            return -1;
        }

	onnxFile.seekg (0, onnxFile.end);
	size_t size = onnxFile.tellg();
	onnxFile.seekg (0, onnxFile.beg);

	auto *buffer = new char[size];
	onnxFile.read(buffer, size);
	onnxFile.close();

    // Define default RetinaNet parameters to use for TRT export
	int batch = 1;
	float score_thresh = 0.02f;
	int top_n = 1000;
    size_t workspace_size =(1ULL << 30);
    float nms_thresh = 0.4;
    int detections_per_im = 85200;
    bool verbose = true;

    // For now, assume we have already done calibration elsewhere
    // if we want to create an INT8 TensorRT engine, so no need
    // to provide calibration files or model name
    vector<string> calibration_files;
    if (argc == 4){
        string calib_dataset_path = "../calibration_data";
        for (const auto& filepath : fs::directory_iterator(calib_dataset_path)){
            calibration_files.emplace_back(filepath.path());
            printf("%s\n", filepath.path().c_str());
        }
    }

    string model_name = "retinaface";
    string calibration_table = (argc == 4) ? string(argv[3]) : "";

    // Use FP16 precision by default, use INT8 if calibration table is provided
    string precision = "FP16";
    if (argc == 4){
        precision = "INT8";
    }

	cout << "Building engine..." << endl;
	auto engine = retinaface::Engine(buffer, size, batch, precision, score_thresh, top_n,
                                     nms_thresh, detections_per_im, calibration_files, model_name,
                                     calibration_table, verbose, workspace_size);
	engine.save(string(argv[2]));


	delete [] buffer;

	return 0;
}
