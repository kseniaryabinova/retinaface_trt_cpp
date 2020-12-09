# RetinaFace в TensorRT

Тут мы конвертим ONNX-модель в TRT-модель с квантизацией до FP16 и INT8

# Немного о TRT

The core of NVIDIA® TensorRT™ is a C++ library that facilitates high-performance inference on NVIDIA graphics processing units (GPUs). It is designed to work in a complementary fashion with training frameworks such as TensorFlow, Caffe, PyTorch, MXNet, etc. It focuses specifically on running an already-trained network quickly and efficiently on a GPU for the purpose of generating a result (a process that is referred to in various places as scoring, detecting, regression, or inference).

## Что можно сделать с помощью TRT?

- Оптимизировать вычислительный граф нейросети
- Реализовывать кастомные слои на GPU
- Квантизовывать веса нейросети 
- И др. Больше можно почитать [здесь](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)

# Как запустить 

1. Установить TensorRT для С++. [TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
2. Сбилдить конвертер
```bash
mkdir build
cd build
cmake ..
make -j4
```
3. Конвертировать ONNX-модель в engine-файл:
```bash
# конвертим ONNX-модель в engine-файл FP16
./export /path/to/onnx-model retinaface.plan

# ИЛИ

# конвертим ONNX-модель в engine-файл INT8
./export /path/to/onnx-model retinaface.plan calibration_table
```   


# Ссылки на источники

Часть кода для TensorRT-оптимизации взята [отсюда](https://github.com/NVIDIA/retinanet-examples)

1. [TRT developer guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
2. [TRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
3. [TRT best practices](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/index.html)
4. [Convert Onnx BERT model to TensorRT (no C+ required!)](https://medium.com/@hemanths933/convert-onnx-bert-model-to-tensorrt-e809276b01b6)
5. [QA inference BERT on TRT](https://aihub.cloud.google.com/p/products%2F86c3b511-b604-48a5-8e19-0de5023ef057) 
6. [BERT Example using the TensorRT C++ API](https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT)

# Дополнительная информация

Время работы сети в TRT
```bash
[0] Took 0.00361883 seconds per inference.
[1] Took 0.00292792 seconds per inference.
[2] Took 0.00291773 seconds per inference.
[3] Took 0.0029212 seconds per inference.
[4] Took 0.00291975 seconds per inference.
[5] Took 0.00292425 seconds per inference.
[6] Took 0.00296232 seconds per inference.
[7] Took 0.00296401 seconds per inference.
[8] Took 0.0029635 seconds per inference.
[9] Took 0.00296121 seconds per inference.
```