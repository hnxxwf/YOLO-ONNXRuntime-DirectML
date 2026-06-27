#ifndef INFER_H
#define INFER_H

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

struct ModelInputInfo
{
    int channels{3};
    ONNXTensorElementDataType elementType{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
    cv::Size size{640, 640};
    bool dynamicSize{false};
};

struct InputTensorData
{
    ONNXTensorElementDataType elementType{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
    std::array<int64_t, 4> shape{1, 3, 0, 0};
    size_t elementCount{0};
    std::vector<float> floatData;
    std::vector<Ort::Float16_t> halfData;
};

struct OutputTensorView
{
    std::vector<int64_t> shape;
    ONNXTensorElementDataType elementType{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
    const void *data{nullptr};
};

class OrtInferEngine
{
public:
    OrtInferEngine();
    ~OrtInferEngine();

    void setModel(const std::string &modelPath);
    const ModelInputInfo &inputInfo() const;
    const std::string &modelPath() const;
    const std::string &deviceName() const;

    std::vector<Ort::Value> infer(InputTensorData &input);
    const std::vector<OutputTensorView> &inferViews(InputTensorData &input);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // INFER_H
