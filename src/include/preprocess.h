#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <array>
#include <cstdint>

#include <opencv2/opencv.hpp>

#include "infer.h"

struct PreprocessResult
{
    cv::Size originalSize;
    cv::Size tensorSize;
    float scale{1.0f};
    float invScale{1.0f};
    float padX{0.0f};
    float padY{0.0f};
};

struct PreprocessBuffers
{
    cv::Mat source;
    cv::Mat resized;
};

struct PreprocessContext
{
    std::array<float, 256> floatLut{};
    std::array<Ort::Float16_t, 256> halfLut{};
    std::array<std::uint16_t, 256> halfBitsLut{};
    PreprocessBuffers buffers;
};

void initPreprocessContext(PreprocessContext &context);

PreprocessResult preprocess(
    const ModelInputInfo &inputInfo,
    const cv::Mat &image,
    PreprocessContext &context,
    InputTensorData &inputTensor);

void preprocess(int modelDimensions, cv::Mat &mat);

#endif // PREPROCESS_H
