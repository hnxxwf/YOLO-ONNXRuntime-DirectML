#include "YoloOrtDml.h"
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <thread>
#include "draw.h"
#include "infer.h"
#include "postprocess.h"
#include "preprocess.h"

namespace
{
int envInt(const char *name, int fallback)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : fallback;
}

int defaultOpenCvThreads()
{
    const unsigned int hardwareThreads = std::thread::hardware_concurrency();
    return static_cast<int>(std::min(4u, hardwareThreads == 0 ? 4u : hardwareThreads));
}
}

struct YoloOrtDml::Pimpl
{
    Pimpl()
    {
        cv::setUseOptimized(true);
        cv::setNumThreads(envInt("YOLO_OPENCV_THREADS", defaultOpenCvThreads()));
        initPreprocessContext(preprocessContext);
    }

    void setModel(const std::string &modelPath)
    {
        engine.setModel(modelPath);
        inputReady = false;
        outputReady = false;
        outputTensors.clear();
        outputViews = nullptr;
        resultBoxes.clear();
    }

    void setConfThreshold(float value)
    {
        confThreshold = value >= 0.0f ? value : 0.4f;
    }

    void setNmsThreshold(float value)
    {
        nmsThreshold = value > 0.0f ? value : 0.45f;
    }

    void setImage(const cv::Mat &mat)
    {
        if (mat.empty()) {
            throw std::invalid_argument("YoloOrtDml received an empty image.");
        }
        if (mat.depth() != CV_8U) {
            throw std::invalid_argument("YoloOrtDml expects an 8-bit OpenCV image.");
        }

        mat.copyTo(image);
        inputReady = false;
        outputReady = false;
        resultBoxes.clear();
    }

    void preprocessImage()
    {
        if (image.empty()) {
            throw std::runtime_error("No image has been provided.");
        }
        if (inputReady) {
            return;
        }

        preprocessInfo = ::preprocess(engine.inputInfo(), image, preprocessContext, inputTensor);
        inputReady = true;
        outputReady = false;
        resultBoxes.clear();
    }

    void runInfer()
    {
        if (!inputReady) {
            preprocessImage();
        }

        outputViews = &engine.inferViews(inputTensor);
        outputReady = true;
        resultBoxes.clear();
    }

    std::vector<DetectResultBox> runPostprocess()
    {
        if (!outputReady) {
            throw std::runtime_error("Model output is not ready.");
        }

        if (outputViews == nullptr) {
            throw std::runtime_error("Model output view is not ready.");
        }

        postprocessYoloOutput(
            *outputViews,
            preprocessInfo,
            confThreshold,
            nmsThreshold,
            postprocessContext,
            resultBoxes);
        return resultBoxes;
    }

    cv::Mat drawResult()
    {
        if (image.empty()) {
            return {};
        }

        cv::Mat result = image.clone();
        ::draw(result, resultBoxes);
        return result;
    }

    OrtInferEngine engine;
    float confThreshold{0.4f};
    float nmsThreshold{0.45f};
    bool inputReady{false};
    bool outputReady{false};

    cv::Mat image;
    PreprocessContext preprocessContext;
    PreprocessResult preprocessInfo;
    InputTensorData inputTensor;
    std::vector<Ort::Value> outputTensors;
    const std::vector<OutputTensorView> *outputViews{nullptr};
    PostprocessContext postprocessContext;
    std::vector<DetectResultBox> resultBoxes;
};

YoloOrtDml::YoloOrtDml()
    : pimpl(std::make_unique<Pimpl>())
{
}

YoloOrtDml::~YoloOrtDml() = default;

void YoloOrtDml::setModel(std::string &modelPath)
{
    pimpl->setModel(modelPath);
}

void YoloOrtDml::setConfThreshold(float confThreshold)
{
    pimpl->setConfThreshold(confThreshold);
}

void YoloOrtDml::setNmsThreshold(float nmsThreshold)
{
    pimpl->setNmsThreshold(nmsThreshold);
}

void YoloOrtDml::preprocess(cv::Mat &mat)
{
    pimpl->setImage(mat);
    pimpl->preprocessImage();
}

void YoloOrtDml::infer()
{
    pimpl->runInfer();
}

std::vector<DetectResultBox> YoloOrtDml::postprocess()
{
    return pimpl->runPostprocess();
}

cv::Mat YoloOrtDml::draw()
{
    return pimpl->drawResult();
}
