#include "YoloOrtDml.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

int parsePositiveInt(const char *text, int fallback)
{
    const int value = std::atoi(text);
    return value > 0 ? value : fallback;
}

void printUsage(const char *program)
{
    std::cout << "Usage:\n"
              << "  " << program << " [model.onnx] [image] [runs] [warmup] [--no-window]\n\n"
              << "Defaults:\n"
              << "  model  = assert/yolov5n_320_fp16.onnx\n"
              << "  image  = assert/cat.png\n"
              << "  runs   = 100\n"
              << "  warmup = 20\n";
}
}

int main(int argc, char **argv)
{
    const std::string defaultModelPath = R"(../../assets/yolov6n_320_fp16.onnx)";
    const std::string defaultImagePath = R"(../../assets/cat.png)";

    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        printUsage(argv[0]);
        return 0;
    }

    std::string modelPath = argc > 1 ? argv[1] : defaultModelPath;
    const std::string imagePath = argc > 2 ? argv[2] : defaultImagePath;
    const int runs = argc > 3 ? parsePositiveInt(argv[3], 100) : 100;
    const int warmupRuns = argc > 4 ? parsePositiveInt(argv[4], 20) : 20;
    const bool showWindow = argc <= 5 || std::string(argv[5]) != "--no-window";

    try {
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << imagePath << std::endl;
            return 1;
        }

        YoloOrtDml detector;
        detector.setModel(modelPath);
        detector.setConfThreshold(0.4f);
        detector.setNmsThreshold(0.45f);

        std::cout << "Image: " << imagePath << " (" << image.cols << "x" << image.rows << ")\n"
                  << "Runs: " << runs << ", warmup: " << warmupRuns << '\n';

        for (int i = 0; i < warmupRuns; ++i) {
            detector.preprocess(image);
            detector.infer();
            detector.postprocess();
        }

        double preprocessTotal = 0.0;
        double inferTotal = 0.0;
        double postprocessTotal = 0.0;
        double totalMin = std::numeric_limits<double>::max();
        double totalMax = 0.0;
        std::vector<DetectResultBox> boxes;

        for (int i = 0; i < runs; ++i) {
            const auto totalBegin = Clock::now();

            const auto preprocessBegin = Clock::now();
            detector.preprocess(image);
            const auto preprocessEnd = Clock::now();

            const auto inferBegin = Clock::now();
            detector.infer();
            const auto inferEnd = Clock::now();

            const auto postprocessBegin = Clock::now();
            boxes = detector.postprocess();
            const auto postprocessEnd = Clock::now();

            const double preprocessMs = elapsedMs(preprocessBegin, preprocessEnd);
            const double inferMs = elapsedMs(inferBegin, inferEnd);
            const double postprocessMs = elapsedMs(postprocessBegin, postprocessEnd);
            const double totalMs = elapsedMs(totalBegin, postprocessEnd);

            preprocessTotal += preprocessMs;
            inferTotal += inferMs;
            postprocessTotal += postprocessMs;
            totalMin = std::min(totalMin, totalMs);
            totalMax = std::max(totalMax, totalMs);
        }

        const double preprocessAvg = preprocessTotal / runs;
        const double inferAvg = inferTotal / runs;
        const double postprocessAvg = postprocessTotal / runs;
        const double totalAvg = preprocessAvg + inferAvg + postprocessAvg;
        const double fps = totalAvg > 0.0 ? 1000.0 / totalAvg : 0.0;

        std::cout << std::fixed << std::setprecision(3)
                  << "\nLatency average:\n"
                  << "  preprocess : " << preprocessAvg << " ms\n"
                  << "  infer      : " << inferAvg << " ms\n"
                  << "  postprocess: " << postprocessAvg << " ms\n"
                  << "  total      : " << totalAvg << " ms (" << fps << " FPS)\n"
                  << "  total min  : " << totalMin << " ms\n"
                  << "  total max  : " << totalMax << " ms\n"
                  << "\nDetections: " << boxes.size() << '\n';

        for (size_t i = 0; i < boxes.size(); ++i) {
            const DetectResultBox &box = boxes[i];
            std::cout << "  [" << i << "] "
                      << box.label << " conf=" << box.confidence
                      << " box=(" << box.x << ", " << box.y << ", "
                      << box.width << ", " << box.height << ")\n";
        }

        cv::Mat result = detector.draw();
        if (!result.empty() && showWindow) {
            cv::imshow("YoloOrtDml Result", result);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    } catch (const std::exception &exception) {
        std::cerr << "YOLO test failed: " << exception.what() << std::endl;
        return 1;
    }

    return 0;
}
