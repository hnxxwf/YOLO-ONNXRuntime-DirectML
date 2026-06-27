#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <emmintrin.h>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace
{
OutputTensorView tensorView(const Ort::Value &tensor)
{
    const auto info = tensor.GetTensorTypeAndShapeInfo();
    OutputTensorView view;
    view.shape = info.GetShape();
    view.elementType = info.GetElementType();

    if (view.elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        view.data = tensor.GetTensorData<float>();
    } else if (view.elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        view.data = tensor.GetTensorData<Ort::Float16_t>();
    } else {
        throw std::runtime_error("Only float32 and float16 model outputs are supported.");
    }

    return view;
}

int clampInt(int value, int low, int high)
{
    if (low > high) {
        std::swap(low, high);
    }
    return std::max(low, std::min(value, high));
}

bool isPositiveSize(const cv::Size &size)
{
    return size.width > 0 && size.height > 0;
}

bool debugPostprocess()
{
    static const bool enabled = []() {
        const char *value = std::getenv("YOLO_POSTPROCESS_DEBUG");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

bool hasObjectness(size_t featureCount)
{
    const size_t cocoClasses = labelsName.size();
    if (featureCount == cocoClasses + 4) {
        return false;
    }
    if (featureCount == cocoClasses + 5) {
        return true;
    }

    return featureCount > cocoClasses + 4;
}

std::string classNameForId(int classId)
{
    if (classId >= 0 && static_cast<size_t>(classId) < labelsName.size()) {
        return labelsName[static_cast<size_t>(classId)];
    }
    return std::to_string(classId);
}

void addResult(
    std::vector<PostprocessCandidate> &boxes,
    int x,
    int y,
    int width,
    int height,
    float score,
    int classId)
{
    if (width <= 0 || height <= 0 || !std::isfinite(score)) {
        return;
    }

    PostprocessCandidate box;
    box.x = x;
    box.y = y;
    box.width = width;
    box.height = height;
    box.confidence = score;
    box.area = static_cast<float>(width * height);
    box.classId = classId;
    boxes.push_back(box);
}

void addCenterResult(
    std::vector<PostprocessCandidate> &boxes,
    const PreprocessResult &preprocess,
    float centerX,
    float centerY,
    float width,
    float height,
    float score,
    int classId)
{
    const float left = (centerX - width * 0.5f - preprocess.padX) * preprocess.invScale;
    const float top = (centerY - height * 0.5f - preprocess.padY) * preprocess.invScale;
    const int x = clampInt(static_cast<int>(left), 0, preprocess.originalSize.width - 1);
    const int y = clampInt(static_cast<int>(top), 0, preprocess.originalSize.height - 1);
    const int boxWidth = clampInt(static_cast<int>(width * preprocess.invScale), 1, preprocess.originalSize.width - x);
    const int boxHeight = clampInt(static_cast<int>(height * preprocess.invScale), 1, preprocess.originalSize.height - y);
    addResult(
        boxes,
        x,
        y,
        boxWidth,
        boxHeight,
        score,
        classId);
}

void addCornerResult(
    std::vector<PostprocessCandidate> &boxes,
    const PreprocessResult &preprocess,
    float x1,
    float y1,
    float x2,
    float y2,
    float score,
    int classId)
{
    const int left = clampInt(static_cast<int>((x1 - preprocess.padX) * preprocess.invScale), 0, preprocess.originalSize.width - 1);
    const int top = clampInt(static_cast<int>((y1 - preprocess.padY) * preprocess.invScale), 0, preprocess.originalSize.height - 1);
    const int right = clampInt(static_cast<int>((x2 - preprocess.padX) * preprocess.invScale), left + 1, preprocess.originalSize.width);
    const int bottom = clampInt(static_cast<int>((y2 - preprocess.padY) * preprocess.invScale), top + 1, preprocess.originalSize.height);
    addResult(
        boxes,
        left,
        top,
        right - left,
        bottom - top,
        score,
        classId);
}

int bestClassFloat(const float *scores, int count)
{
    int bestClass = 0;
    float bestScore = scores[0];
    for (int c = 1; c < count; ++c) {
        if (scores[c] > bestScore) {
            bestScore = scores[c];
            bestClass = c;
        }
    }
    return bestClass;
}

int bestClassHalf(const Ort::Float16_t *scores, int count)
{
    int bestClass = 0;
    uint16_t bestBits = scores[0].val;
    for (int c = 1; c < count; ++c) {
        const uint16_t bits = scores[c].val;
        if (bits > bestBits) {
            bestBits = bits;
            bestClass = c;
        }
    }
    return bestClass;
}

void computeBestFloatScoresPlanar(
    const float *classData,
    int classCount,
    size_t boxCount,
    std::vector<float> &bestScores)
{
    bestScores.resize(boxCount);
    if (boxCount == 0) {
        return;
    }

    std::memcpy(bestScores.data(), classData, boxCount * sizeof(float));
    float *best = bestScores.data();
    for (int c = 1; c < classCount; ++c) {
        const float *scores = classData + static_cast<size_t>(c) * boxCount;
        size_t i = 0;
        for (; i + 3 < boxCount; i += 4) {
            const __m128 bestValues = _mm_loadu_ps(best + i);
            const __m128 scoreValues = _mm_loadu_ps(scores + i);
            _mm_storeu_ps(best + i, _mm_max_ps(bestValues, scoreValues));
        }
        for (; i < boxCount; ++i) {
            if (scores[i] > best[i]) {
                best[i] = scores[i];
            }
        }
    }
}

void computeBestHalfScoresPlanar(
    const Ort::Float16_t *classData,
    int classCount,
    size_t boxCount,
    std::vector<std::uint16_t> &bestScores)
{
    bestScores.resize(boxCount);
    if (boxCount == 0) {
        return;
    }

    std::memcpy(bestScores.data(), classData, boxCount * sizeof(std::uint16_t));
    std::uint16_t *best = bestScores.data();
    for (int c = 1; c < classCount; ++c) {
        const auto *scores = reinterpret_cast<const std::uint16_t *>(
            classData + static_cast<size_t>(c) * boxCount);
        size_t i = 0;
        for (; i + 7 < boxCount; i += 8) {
            const __m128i bestValues = _mm_loadu_si128(reinterpret_cast<const __m128i *>(best + i));
            const __m128i scoreValues = _mm_loadu_si128(reinterpret_cast<const __m128i *>(scores + i));
            const __m128i mask = _mm_cmpgt_epi16(scoreValues, bestValues);
            const __m128i selected = _mm_or_si128(
                _mm_and_si128(mask, scoreValues),
                _mm_andnot_si128(mask, bestValues));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(best + i), selected);
        }
        for (; i < boxCount; ++i) {
            if (scores[i] > best[i]) {
                best[i] = scores[i];
            }
        }
    }
}

void collectRowMajor(
    const OutputTensorView &output,
    const PreprocessResult &preprocess,
    float confThreshold,
    std::vector<PostprocessCandidate> &boxes)
{
    const size_t boxCount = static_cast<size_t>(output.shape[1]);
    const size_t features = static_cast<size_t>(output.shape[2]);
    if (features <= 4 || boxCount == 0) {
        return;
    }

    const bool hasObjectScore = hasObjectness(features);
    const size_t classOffset = hasObjectScore ? 5 : 4;
    const int classCount = static_cast<int>(features - classOffset);
    if (classCount <= 0) {
        return;
    }

    boxes.reserve(std::max<size_t>(boxes.capacity(), 256));

    if (output.elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float *values = static_cast<const float *>(output.data);
        for (size_t i = 0; i < boxCount; ++i) {
            const float *row = values + i * features;
            const float objectScore = hasObjectScore ? row[4] : 1.0f;
            if (objectScore <= confThreshold) {
                continue;
            }

            const int classId = bestClassFloat(row + classOffset, classCount);
            const float score = objectScore * row[classOffset + static_cast<size_t>(classId)];
            if (score <= confThreshold) {
                continue;
            }

            addCenterResult(boxes, preprocess, row[0], row[1], row[2], row[3], score, classId);
        }
    } else {
        const Ort::Float16_t *values = static_cast<const Ort::Float16_t *>(output.data);
        const uint16_t confBits = Ort::Float16_t(confThreshold).val;
        const uint16_t oneBits = Ort::Float16_t(1.0f).val;
        for (size_t i = 0; i < boxCount; ++i) {
            const Ort::Float16_t *row = values + i * features;
            const uint16_t objectBits = hasObjectScore ? row[4].val : oneBits;
            if (objectBits <= confBits) {
                continue;
            }

            const int classId = bestClassHalf(row + classOffset, classCount);
            const float objectScore = hasObjectScore ? row[4].ToFloat() : 1.0f;
            const float score = objectScore * row[classOffset + static_cast<size_t>(classId)].ToFloat();
            if (score <= confThreshold) {
                continue;
            }

            addCenterResult(
                boxes,
                preprocess,
                row[0].ToFloat(),
                row[1].ToFloat(),
                row[2].ToFloat(),
                row[3].ToFloat(),
                score,
                classId);
        }
    }
}

void collectPlanar(
    const OutputTensorView &output,
    const PreprocessResult &preprocess,
    float confThreshold,
    PostprocessContext &context)
{
    const size_t features = static_cast<size_t>(output.shape[1]);
    const size_t boxCount = static_cast<size_t>(output.shape[2]);
    if (features <= 4 || boxCount == 0) {
        return;
    }

    const bool hasObjectScore = hasObjectness(features);
    const size_t classOffset = hasObjectScore ? 5 : 4;
    const int classCount = static_cast<int>(features - classOffset);
    if (classCount <= 0) {
        return;
    }

    std::vector<PostprocessCandidate> &boxes = context.candidates;
    boxes.reserve(std::max<size_t>(boxes.capacity(), 256));

    if (output.elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float *values = static_cast<const float *>(output.data);
        const float *xData = values;
        const float *yData = values + boxCount;
        const float *wData = values + 2 * boxCount;
        const float *hData = values + 3 * boxCount;
        const float *objectData = hasObjectScore ? values + 4 * boxCount : nullptr;
        const float *classData = values + classOffset * boxCount;

        if (objectData == nullptr) {
            computeBestFloatScoresPlanar(classData, classCount, boxCount, context.bestFloatScores);

            for (size_t i = 0; i < boxCount; ++i) {
                const float score = context.bestFloatScores[i];
                if (score <= confThreshold) {
                    continue;
                }
                int classId = 0;
                if (classData[i] != score) {
                    for (int c = 1; c < classCount; ++c) {
                        if (classData[static_cast<size_t>(c) * boxCount + i] == score) {
                            classId = c;
                            break;
                        }
                    }
                }

                addCenterResult(
                    boxes,
                    preprocess,
                    xData[i],
                    yData[i],
                    wData[i],
                    hData[i],
                    score,
                    classId);
            }
            return;
        }

        for (size_t i = 0; i < boxCount; ++i) {
            const float objectScore = objectData ? objectData[i] : 1.0f;
            if (objectScore <= confThreshold) {
                continue;
            }

            int classId = 0;
            float bestClassScore = classData[i];
            for (int c = 1; c < classCount; ++c) {
                const float classScore = classData[static_cast<size_t>(c) * boxCount + i];
                if (classScore > bestClassScore) {
                    bestClassScore = classScore;
                    classId = c;
                }
            }

            const float score = objectScore * bestClassScore;
            if (score <= confThreshold) {
                continue;
            }

            addCenterResult(boxes, preprocess, xData[i], yData[i], wData[i], hData[i], score, classId);
        }
    } else {
        const Ort::Float16_t *values = static_cast<const Ort::Float16_t *>(output.data);
        const Ort::Float16_t *xData = values;
        const Ort::Float16_t *yData = values + boxCount;
        const Ort::Float16_t *wData = values + 2 * boxCount;
        const Ort::Float16_t *hData = values + 3 * boxCount;
        const Ort::Float16_t *objectData = hasObjectScore ? values + 4 * boxCount : nullptr;
        const Ort::Float16_t *classData = values + classOffset * boxCount;
        const uint16_t confBits = Ort::Float16_t(confThreshold).val;
        const uint16_t oneBits = Ort::Float16_t(1.0f).val;

        if (objectData == nullptr) {
            computeBestHalfScoresPlanar(classData, classCount, boxCount, context.bestHalfScores);

            for (size_t i = 0; i < boxCount; ++i) {
                const uint16_t scoreBits = context.bestHalfScores[i];
                if (scoreBits <= confBits) {
                    continue;
                }
                int classId = 0;
                if (classData[i].val != scoreBits) {
                    for (int c = 1; c < classCount; ++c) {
                        if (classData[static_cast<size_t>(c) * boxCount + i].val == scoreBits) {
                            classId = c;
                            break;
                        }
                    }
                }

                Ort::Float16_t scoreValue;
                scoreValue.val = scoreBits;
                addCenterResult(
                    boxes,
                    preprocess,
                    xData[i].ToFloat(),
                    yData[i].ToFloat(),
                    wData[i].ToFloat(),
                    hData[i].ToFloat(),
                    scoreValue.ToFloat(),
                    classId);
            }
            return;
        }

        for (size_t i = 0; i < boxCount; ++i) {
            const uint16_t objectBits = objectData ? objectData[i].val : oneBits;
            if (objectBits <= confBits) {
                continue;
            }

            int classId = 0;
            uint16_t bestClassBits = classData[i].val;
            for (int c = 1; c < classCount; ++c) {
                const uint16_t classBits = classData[static_cast<size_t>(c) * boxCount + i].val;
                if (classBits > bestClassBits) {
                    bestClassBits = classBits;
                    classId = c;
                }
            }

            const float objectScore = objectData ? objectData[i].ToFloat() : 1.0f;
            const float bestClassScore = classData[static_cast<size_t>(classId) * boxCount + i].ToFloat();
            const float score = objectScore * bestClassScore;
            if (score <= confThreshold) {
                continue;
            }

            addCenterResult(
                boxes,
                preprocess,
                xData[i].ToFloat(),
                yData[i].ToFloat(),
                wData[i].ToFloat(),
                hData[i].ToFloat(),
                score,
                classId);
        }
    }
}

void collectEndToEnd(
    const OutputTensorView &output,
    const PreprocessResult &preprocess,
    float confThreshold,
    std::vector<PostprocessCandidate> &boxes)
{
    const size_t boxCount = static_cast<size_t>(output.shape[1]);
    if (boxCount == 0) {
        return;
    }

    if (output.elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float *values = static_cast<const float *>(output.data);
        for (size_t i = 0; i < boxCount; ++i) {
            const float *row = values + i * 6;
            if (row[4] <= confThreshold) {
                continue;
            }
            addCornerResult(boxes, preprocess, row[0], row[1], row[2], row[3], row[4], static_cast<int>(row[5]));
        }
    } else {
        const Ort::Float16_t *values = static_cast<const Ort::Float16_t *>(output.data);
        const uint16_t confBits = Ort::Float16_t(confThreshold).val;
        for (size_t i = 0; i < boxCount; ++i) {
            const Ort::Float16_t *row = values + i * 6;
            if (row[4].val <= confBits) {
                continue;
            }
            addCornerResult(
                boxes,
                preprocess,
                row[0].ToFloat(),
                row[1].ToFloat(),
                row[2].ToFloat(),
                row[3].ToFloat(),
                row[4].ToFloat(),
                static_cast<int>(row[5].ToFloat()));
        }
    }
}

float intersectionOverUnion(const PostprocessCandidate &lhs, const PostprocessCandidate &rhs)
{
    const int x1 = std::max(lhs.x, rhs.x);
    const int y1 = std::max(lhs.y, rhs.y);
    const int x2 = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
    const int y2 = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
    const int width = x2 - x1;
    const int height = y2 - y1;
    if (width <= 0 || height <= 0) {
        return 0.0f;
    }

    const float intersection = static_cast<float>(width * height);
    const float unionArea = lhs.area + rhs.area - intersection;
    return unionArea > 0.0f ? intersection / unionArea : 0.0f;
}

void nmsCandidates(
    float confThreshold,
    float nmsThreshold,
    const std::vector<PostprocessCandidate> &candidates,
    PostprocessContext &context)
{
    if (confThreshold < 0.0f) {
        confThreshold = 0.4f;
    }
    if (nmsThreshold <= 0.0f) {
        nmsThreshold = 0.45f;
    }

    if (candidates.empty()) {
        context.kept.clear();
        return;
    }
    if (candidates.size() == 1) {
        context.kept.assign(1, 0);
        return;
    }

    context.order.resize(candidates.size());
    std::iota(context.order.begin(), context.order.end(), 0);
    std::sort(context.order.begin(), context.order.end(), [&candidates](int lhs, int rhs) {
        return candidates[static_cast<size_t>(lhs)].confidence
            > candidates[static_cast<size_t>(rhs)].confidence;
    });

    context.suppressed.assign(candidates.size(), 0);
    context.kept.clear();
    context.kept.reserve(candidates.size());

    for (size_t orderIndex = 0; orderIndex < context.order.size(); ++orderIndex) {
        const int currentIndex = context.order[orderIndex];
        if (context.suppressed[static_cast<size_t>(currentIndex)] != 0) {
            continue;
        }

        const PostprocessCandidate &current = candidates[static_cast<size_t>(currentIndex)];
        if (current.confidence < confThreshold) {
            break;
        }
        context.kept.push_back(currentIndex);

        for (size_t nextIndex = orderIndex + 1; nextIndex < context.order.size(); ++nextIndex) {
            const int candidateIndex = context.order[nextIndex];
            if (context.suppressed[static_cast<size_t>(candidateIndex)] != 0) {
                continue;
            }

            const PostprocessCandidate &candidate = candidates[static_cast<size_t>(candidateIndex)];
            if (candidate.confidence < confThreshold) {
                break;
            }
            if (candidate.classId == current.classId
                && intersectionOverUnion(current, candidate) > nmsThreshold) {
                context.suppressed[static_cast<size_t>(candidateIndex)] = 1;
            }
        }
    }
}

void materializeResults(
    const std::vector<PostprocessCandidate> &candidates,
    const std::vector<int> &kept,
    std::vector<DetectResultBox> &results)
{
    results.clear();
    results.reserve(kept.size());
    for (const int index : kept) {
        const PostprocessCandidate &candidate = candidates[static_cast<size_t>(index)];
        DetectResultBox box;
        box.x = candidate.x;
        box.y = candidate.y;
        box.width = candidate.width;
        box.height = candidate.height;
        box.confidence = candidate.confidence;
        box.classId = candidate.classId;
        box.label = classNameForId(candidate.classId);
        results.push_back(std::move(box));
    }
}

PostprocessContext makeStackContext()
{
    PostprocessContext context;
    context.candidates.reserve(256);
    context.results.reserve(256);
    return context;
}

void nmsDetectResultBoxes(
    float confThreshold,
    float nmsThreshold,
    std::vector<DetectResultBox> &detectResultBoxes)
{
    if (confThreshold < 0.0f) {
        confThreshold = 0.4f;
    }
    if (nmsThreshold <= 0.0f) {
        nmsThreshold = 0.45f;
    }

    PostprocessContext context = makeStackContext();
    context.candidates.clear();
    context.candidates.reserve(detectResultBoxes.size());

    for (const DetectResultBox &box : detectResultBoxes) {
        if (box.width <= 0.0f
            || box.height <= 0.0f
            || !std::isfinite(box.confidence)
            || box.confidence < confThreshold) {
            continue;
        }

        PostprocessCandidate candidate;
        candidate.x = box.x;
        candidate.y = box.y;
        candidate.width = box.width;
        candidate.height = box.height;
        candidate.confidence = box.confidence;
        candidate.area = box.width * box.height;
        candidate.classId = box.classId;
        context.candidates.push_back(candidate);
    }

    nmsCandidates(confThreshold, nmsThreshold, context.candidates, context);
    materializeResults(context.candidates, context.kept, detectResultBoxes);
}

void postprocessViews(
    const std::vector<OutputTensorView> &outputTensors,
    const PreprocessResult &preprocess,
    float confThreshold,
    float nmsThreshold,
    PostprocessContext &context,
    std::vector<DetectResultBox> &results)
{
    context.candidates.clear();
    if (outputTensors.empty() || !isPositiveSize(preprocess.originalSize)) {
        results.clear();
        return;
    }

    const OutputTensorView &output = outputTensors[0];
    if (output.shape.size() < 3 || output.data == nullptr) {
        results.clear();
        return;
    }

    if (output.shape[2] == 6) {
        collectEndToEnd(output, preprocess, confThreshold, context.candidates);
    } else if (output.shape[1] > output.shape[2]) {
        collectRowMajor(output, preprocess, confThreshold, context.candidates);
    } else {
        collectPlanar(output, preprocess, confThreshold, context);
    }

    nmsCandidates(confThreshold, nmsThreshold, context.candidates, context);
    if (debugPostprocess()) {
        std::cerr << "[YoloOrtDml] postprocess shape=[";
        for (size_t i = 0; i < output.shape.size(); ++i) {
            if (i != 0) {
                std::cerr << ',';
            }
            std::cerr << output.shape[i];
        }
        std::cerr << "] candidates=" << context.candidates.size()
                  << " kept=" << context.kept.size() << std::endl;
    }
    materializeResults(context.candidates, context.kept, results);
}
}

std::vector<DetectResultBox> postprocessYoloOutput(
    const std::vector<Ort::Value> &outputTensors,
    const PreprocessResult &preprocess,
    float confThreshold,
    float nmsThreshold,
    PostprocessContext &context)
{
    postprocessYoloOutput(outputTensors, preprocess, confThreshold, nmsThreshold, context, context.results);
    return context.results;
}

std::vector<DetectResultBox> postprocessYoloOutput(
    const std::vector<OutputTensorView> &outputTensors,
    const PreprocessResult &preprocess,
    float confThreshold,
    float nmsThreshold,
    PostprocessContext &context)
{
    postprocessYoloOutput(outputTensors, preprocess, confThreshold, nmsThreshold, context, context.results);
    return context.results;
}

void postprocessYoloOutput(
    const std::vector<Ort::Value> &outputTensors,
    const PreprocessResult &preprocess,
    float confThreshold,
    float nmsThreshold,
    PostprocessContext &context,
    std::vector<DetectResultBox> &results)
{
    context.results.clear();
    context.results.reserve(outputTensors.size());
    std::vector<OutputTensorView> views;
    views.reserve(outputTensors.size());
    for (const Ort::Value &tensor : outputTensors) {
        views.push_back(tensorView(tensor));
    }

    postprocessViews(views, preprocess, confThreshold, nmsThreshold, context, results);
}

void postprocessYoloOutput(
    const std::vector<OutputTensorView> &outputTensors,
    const PreprocessResult &preprocess,
    float confThreshold,
    float nmsThreshold,
    PostprocessContext &context,
    std::vector<DetectResultBox> &results)
{
    postprocessViews(outputTensors, preprocess, confThreshold, nmsThreshold, context, results);
}

void postprocess(float confThreshold, float nmsThreshold, std::vector<DetectResultBox> &detectResultBoxes)
{
    nmsDetectResultBoxes(confThreshold, nmsThreshold, detectResultBoxes);
}
