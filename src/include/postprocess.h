#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <cstdint>
#include <vector>

#include "model.h"
#include "preprocess.h"

struct PostprocessCandidate
{
    int x{0};
    int y{0};
    int width{0};
    int height{0};
    float confidence{0.0f};
    float area{0.0f};
    int classId{-1};
};

struct PostprocessContext
{
    std::vector<int> order;
    std::vector<unsigned char> suppressed;
    std::vector<int> kept;
    std::vector<PostprocessCandidate> candidates;
    std::vector<DetectResultBox> results;
    std::vector<int> bestClassIds;
    std::vector<float> bestFloatScores;
    std::vector<std::uint16_t> bestHalfScores;
};

std::vector<DetectResultBox> postprocessYoloOutput(
    const std::vector<Ort::Value> &outputTensors,
    const PreprocessResult &preprocess,
    float confThreshold,
    float nmsThreshold,
    PostprocessContext &context);

std::vector<DetectResultBox> postprocessYoloOutput(
    const std::vector<OutputTensorView> &outputTensors,
    const PreprocessResult &preprocess,
    float confThreshold,
    float nmsThreshold,
    PostprocessContext &context);

void postprocessYoloOutput(
    const std::vector<Ort::Value> &outputTensors,
    const PreprocessResult &preprocess,
    float confThreshold,
    float nmsThreshold,
    PostprocessContext &context,
    std::vector<DetectResultBox> &results);

void postprocessYoloOutput(
    const std::vector<OutputTensorView> &outputTensors,
    const PreprocessResult &preprocess,
    float confThreshold,
    float nmsThreshold,
    PostprocessContext &context,
    std::vector<DetectResultBox> &results);

void postprocess(float confThreshold, float nmsThreshold, std::vector<DetectResultBox>& detectResultBoxes);

#endif // POSTPROCESS_H
