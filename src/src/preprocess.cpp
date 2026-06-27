#include "preprocess.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{
int alignUp(int value, int alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

cv::Mat prepareSourceImage(const cv::Mat &image, int inputChannels, cv::Mat &sourceBuffer)
{
    if (inputChannels == 1) {
        if (image.channels() == 1) {
            return image;
        }
        if (image.channels() == 3) {
            cv::cvtColor(image, sourceBuffer, cv::COLOR_BGR2GRAY);
            return sourceBuffer;
        }
        if (image.channels() == 4) {
            cv::cvtColor(image, sourceBuffer, cv::COLOR_BGRA2GRAY);
            return sourceBuffer;
        }
    } else {
        if (image.channels() == 3) {
            return image;
        }
        if (image.channels() == 4) {
            cv::cvtColor(image, sourceBuffer, cv::COLOR_BGRA2BGR);
            return sourceBuffer;
        }
        if (image.channels() == 1) {
            cv::cvtColor(image, sourceBuffer, cv::COLOR_GRAY2BGR);
            return sourceBuffer;
        }
    }

    throw std::invalid_argument("Unsupported image channel count.");
}

struct PreparedImage
{
    cv::Mat source;
    cv::Mat resized;
    PreprocessResult result;
    int tensorWidth{0};
    int tensorHeight{0};
    int resizedWidth{0};
    int resizedHeight{0};
    int padLeft{0};
    int padTop{0};
    bool hasPadding{false};
};

PreparedImage prepareImage(
    cv::Size modelSize,
    bool dynamicInputSize,
    int inputChannels,
    const cv::Mat &image,
    cv::Mat &sourceBuffer,
    cv::Mat &resizedBuffer)
{
    if (image.empty()) {
        throw std::invalid_argument("preprocess received an empty image.");
    }
    if (image.depth() != CV_8U) {
        throw std::invalid_argument("preprocess expects an 8-bit OpenCV image.");
    }
    if (modelSize.width <= 0 || modelSize.height <= 0) {
        throw std::invalid_argument("preprocess received an invalid model input size.");
    }
    if (inputChannels != 1 && inputChannels != 3) {
        throw std::invalid_argument("preprocess supports only 1-channel and 3-channel model inputs.");
    }

    PreparedImage prepared;
    prepared.source = prepareSourceImage(image, inputChannels, sourceBuffer);
    prepared.result.originalSize = cv::Size(prepared.source.cols, prepared.source.rows);
    prepared.result.scale = std::min(
        static_cast<float>(modelSize.height) / static_cast<float>(prepared.source.rows),
        static_cast<float>(modelSize.width) / static_cast<float>(prepared.source.cols));
    prepared.result.invScale = 1.0f / prepared.result.scale;

    prepared.resizedWidth = std::max(1, static_cast<int>(std::round(prepared.source.cols * prepared.result.scale)));
    prepared.resizedHeight = std::max(1, static_cast<int>(std::round(prepared.source.rows * prepared.result.scale)));
    prepared.tensorWidth = modelSize.width;
    prepared.tensorHeight = modelSize.height;
    if (dynamicInputSize) {
        prepared.tensorWidth = std::max(32, alignUp(prepared.resizedWidth, 32));
        prepared.tensorHeight = std::max(32, alignUp(prepared.resizedHeight, 32));
    }

    prepared.result.tensorSize = cv::Size(prepared.tensorWidth, prepared.tensorHeight);
    prepared.result.padX = static_cast<float>(prepared.tensorWidth - prepared.resizedWidth) * 0.5f;
    prepared.result.padY = static_cast<float>(prepared.tensorHeight - prepared.resizedHeight) * 0.5f;
    prepared.padLeft = static_cast<int>(std::round(prepared.result.padX - 0.1f));
    prepared.padTop = static_cast<int>(std::round(prepared.result.padY - 0.1f));
    prepared.hasPadding = prepared.resizedWidth != prepared.tensorWidth
        || prepared.resizedHeight != prepared.tensorHeight
        || prepared.padLeft != 0
        || prepared.padTop != 0;

    if (prepared.resizedWidth == prepared.source.cols && prepared.resizedHeight == prepared.source.rows) {
        prepared.resized = prepared.source;
    } else {
        resizedBuffer.create(prepared.resizedHeight, prepared.resizedWidth, prepared.source.type());
        cv::resize(prepared.source, resizedBuffer, resizedBuffer.size(), 0.0, 0.0, cv::INTER_LINEAR);
        prepared.resized = resizedBuffer;
    }

    return prepared;
}

void setTensorShape(
    int tensorWidth,
    int tensorHeight,
    int inputChannels,
    std::array<int64_t, 4> &inputShape,
    size_t &inputElementCount)
{
    inputElementCount = static_cast<size_t>(tensorWidth)
        * static_cast<size_t>(tensorHeight)
        * static_cast<size_t>(inputChannels);
    inputShape = {
        1,
        static_cast<int64_t>(inputChannels),
        static_cast<int64_t>(tensorHeight),
        static_cast<int64_t>(tensorWidth)
    };
}

void writeFloatInput(
    const PreparedImage &prepared,
    int inputChannels,
    const std::array<float, 256> &inputLut,
    std::vector<float> &inputTensorData)
{
    if (prepared.hasPadding) {
        std::fill(inputTensorData.begin(), inputTensorData.end(), inputLut[114]);
    }

    if (inputChannels == 1) {
        for (int y = 0; y < prepared.resized.rows; ++y) {
            const int dstOffset = (y + prepared.padTop) * prepared.tensorWidth + prepared.padLeft;
            const uint8_t *src = prepared.resized.ptr<uint8_t>(y);
            float *dst = inputTensorData.data() + dstOffset;
            for (int x = 0; x < prepared.resized.cols; ++x) {
                dst[x] = inputLut[src[x]];
            }
        }
        return;
    }

    const size_t area = static_cast<size_t>(prepared.tensorWidth) * static_cast<size_t>(prepared.tensorHeight);
    float *red = inputTensorData.data();
    float *green = red + area;
    float *blue = green + area;

    if (!prepared.hasPadding && prepared.resized.isContinuous()) {
        const uint8_t *src = prepared.resized.ptr<uint8_t>(0);
        float *dstR = red;
        float *dstG = green;
        float *dstB = blue;
        const size_t pixelCount = static_cast<size_t>(prepared.resized.cols) * static_cast<size_t>(prepared.resized.rows);
        size_t i = 0;
        for (; i + 3 < pixelCount; i += 4) {
            dstR[0] = inputLut[src[2]];
            dstG[0] = inputLut[src[1]];
            dstB[0] = inputLut[src[0]];
            dstR[1] = inputLut[src[5]];
            dstG[1] = inputLut[src[4]];
            dstB[1] = inputLut[src[3]];
            dstR[2] = inputLut[src[8]];
            dstG[2] = inputLut[src[7]];
            dstB[2] = inputLut[src[6]];
            dstR[3] = inputLut[src[11]];
            dstG[3] = inputLut[src[10]];
            dstB[3] = inputLut[src[9]];
            src += 12;
            dstR += 4;
            dstG += 4;
            dstB += 4;
        }
        for (; i < pixelCount; ++i) {
            *dstR++ = inputLut[src[2]];
            *dstG++ = inputLut[src[1]];
            *dstB++ = inputLut[src[0]];
            src += 3;
        }
        return;
    }

    for (int y = 0; y < prepared.resized.rows; ++y) {
        const int dstOffset = (y + prepared.padTop) * prepared.tensorWidth + prepared.padLeft;
        const uint8_t *src = prepared.resized.ptr<uint8_t>(y);
        float *dstR = red + dstOffset;
        float *dstG = green + dstOffset;
        float *dstB = blue + dstOffset;
        for (int x = 0; x < prepared.resized.cols; ++x) {
            *dstR++ = inputLut[src[2]];
            *dstG++ = inputLut[src[1]];
            *dstB++ = inputLut[src[0]];
            src += 3;
        }
    }
}

void writeHalfInputNoPadding(
    const PreparedImage &prepared,
    const std::array<std::uint16_t, 256> &inputLut,
    Ort::Float16_t *red,
    Ort::Float16_t *green,
    Ort::Float16_t *blue)
{
    static_assert(sizeof(Ort::Float16_t) == sizeof(uint16_t), "Ort::Float16_t must be 16-bit.");

    const uint8_t *src = prepared.resized.ptr<uint8_t>(0);
    const size_t pixelCount = static_cast<size_t>(prepared.resized.cols) * static_cast<size_t>(prepared.resized.rows);
    const uint16_t *lut = inputLut.data();

    uint16_t *dstR = reinterpret_cast<uint16_t *>(red);
    uint16_t *dstG = reinterpret_cast<uint16_t *>(green);
    uint16_t *dstB = reinterpret_cast<uint16_t *>(blue);
    size_t i = 0;
    for (; i + 7 < pixelCount; i += 8) {
        dstR[0] = lut[src[2]];
        dstG[0] = lut[src[1]];
        dstB[0] = lut[src[0]];
        dstR[1] = lut[src[5]];
        dstG[1] = lut[src[4]];
        dstB[1] = lut[src[3]];
        dstR[2] = lut[src[8]];
        dstG[2] = lut[src[7]];
        dstB[2] = lut[src[6]];
        dstR[3] = lut[src[11]];
        dstG[3] = lut[src[10]];
        dstB[3] = lut[src[9]];
        dstR[4] = lut[src[14]];
        dstG[4] = lut[src[13]];
        dstB[4] = lut[src[12]];
        dstR[5] = lut[src[17]];
        dstG[5] = lut[src[16]];
        dstB[5] = lut[src[15]];
        dstR[6] = lut[src[20]];
        dstG[6] = lut[src[19]];
        dstB[6] = lut[src[18]];
        dstR[7] = lut[src[23]];
        dstG[7] = lut[src[22]];
        dstB[7] = lut[src[21]];
        src += 24;
        dstR += 8;
        dstG += 8;
        dstB += 8;
    }
    for (; i < pixelCount; ++i) {
        *dstR++ = lut[src[2]];
        *dstG++ = lut[src[1]];
        *dstB++ = lut[src[0]];
        src += 3;
    }
}

void writeHalfInput(
    const PreparedImage &prepared,
    int inputChannels,
    const std::array<Ort::Float16_t, 256> &inputLut,
    const std::array<std::uint16_t, 256> &inputBitsLut,
    std::vector<Ort::Float16_t> &inputTensorData)
{
    if (prepared.hasPadding) {
        std::fill(inputTensorData.begin(), inputTensorData.end(), inputLut[114]);
    }

    if (inputChannels == 1) {
        for (int y = 0; y < prepared.resized.rows; ++y) {
            const int dstOffset = (y + prepared.padTop) * prepared.tensorWidth + prepared.padLeft;
            const uint8_t *src = prepared.resized.ptr<uint8_t>(y);
            Ort::Float16_t *dst = inputTensorData.data() + dstOffset;
            for (int x = 0; x < prepared.resized.cols; ++x) {
                dst[x] = inputLut[src[x]];
            }
        }
        return;
    }

    const size_t area = static_cast<size_t>(prepared.tensorWidth) * static_cast<size_t>(prepared.tensorHeight);
    Ort::Float16_t *red = inputTensorData.data();
    Ort::Float16_t *green = red + area;
    Ort::Float16_t *blue = green + area;

    if (!prepared.hasPadding && prepared.resized.isContinuous()) {
        writeHalfInputNoPadding(prepared, inputBitsLut, red, green, blue);
        return;
    }

    for (int y = 0; y < prepared.resized.rows; ++y) {
        const int dstOffset = (y + prepared.padTop) * prepared.tensorWidth + prepared.padLeft;
        const uint8_t *src = prepared.resized.ptr<uint8_t>(y);
        Ort::Float16_t *dstR = red + dstOffset;
        Ort::Float16_t *dstG = green + dstOffset;
        Ort::Float16_t *dstB = blue + dstOffset;
        for (int x = 0; x < prepared.resized.cols; ++x) {
            *dstR++ = inputLut[src[2]];
            *dstG++ = inputLut[src[1]];
            *dstB++ = inputLut[src[0]];
            src += 3;
        }
    }
}

void preprocessImage(
    int modelDimensions,
    bool dynamicInputSize,
    int inputChannels,
    const cv::Mat &image,
    cv::Mat &sourceBuffer,
    cv::Mat &resizedBuffer,
    cv::Mat &preprocessed,
    cv::Size &originalSize,
    cv::Size &tensorSize,
    float &scale,
    float &invScale,
    float &padX,
    float &padY)
{
    const PreparedImage prepared = prepareImage(
        cv::Size(modelDimensions, modelDimensions),
        dynamicInputSize,
        inputChannels,
        image,
        sourceBuffer,
        resizedBuffer);

    originalSize = prepared.result.originalSize;
    tensorSize = prepared.result.tensorSize;
    scale = prepared.result.scale;
    invScale = prepared.result.invScale;
    padX = prepared.result.padX;
    padY = prepared.result.padY;

    if (prepared.hasPadding) {
        preprocessed.create(prepared.tensorHeight, prepared.tensorWidth, prepared.resized.type());
        preprocessed.setTo(cv::Scalar(114, 114, 114, 114));
        prepared.resized.copyTo(preprocessed(cv::Rect(
            prepared.padLeft,
            prepared.padTop,
            prepared.resized.cols,
            prepared.resized.rows)));
    } else {
        preprocessed = prepared.resized;
    }
}
}

void initPreprocessContext(PreprocessContext &context)
{
    constexpr float normalize = 1.0f / 255.0f;
    for (size_t i = 0; i < context.floatLut.size(); ++i) {
        const float value = static_cast<float>(i) * normalize;
        context.floatLut[i] = value;
        context.halfLut[i] = Ort::Float16_t(value);
        context.halfBitsLut[i] = context.halfLut[i].val;
    }
}

PreprocessResult preprocess(
    const ModelInputInfo &inputInfo,
    const cv::Mat &image,
    PreprocessContext &context,
    InputTensorData &inputTensor)
{
    const PreparedImage prepared = prepareImage(
        inputInfo.size,
        inputInfo.dynamicSize,
        inputInfo.channels,
        image,
        context.buffers.source,
        context.buffers.resized);

    inputTensor.elementType = inputInfo.elementType;
    setTensorShape(
        prepared.tensorWidth,
        prepared.tensorHeight,
        inputInfo.channels,
        inputTensor.shape,
        inputTensor.elementCount);

    if (inputInfo.elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        inputTensor.floatData.resize(inputTensor.elementCount);
        writeFloatInput(prepared, inputInfo.channels, context.floatLut, inputTensor.floatData);
    } else if (inputInfo.elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        inputTensor.halfData.resize(inputTensor.elementCount);
        writeHalfInput(prepared, inputInfo.channels, context.halfLut, context.halfBitsLut, inputTensor.halfData);
    } else {
        throw std::runtime_error("Only float32 and float16 model inputs are supported.");
    }

    return prepared.result;
}

void preprocess(int modelDimensions, cv::Mat &mat)
{
    cv::Mat sourceBuffer;
    cv::Mat resizedBuffer;
    cv::Mat preprocessed;
    cv::Size originalSize;
    cv::Size tensorSize;
    float scale = 1.0f;
    float invScale = 1.0f;
    float padX = 0.0f;
    float padY = 0.0f;
    preprocessImage(
        modelDimensions,
        false,
        mat.channels() == 1 ? 1 : 3,
        mat,
        sourceBuffer,
        resizedBuffer,
        preprocessed,
        originalSize,
        tensorSize,
        scale,
        invScale,
        padX,
        padY);
    mat = preprocessed.clone();
}
