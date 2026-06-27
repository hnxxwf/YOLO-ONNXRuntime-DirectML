#include "draw.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>

#include <opencv2/imgproc.hpp>

namespace
{
cv::Scalar colorForClass(int classId)
{
    if (classId >= 0 && static_cast<size_t>(classId) < labelsColor.size()) {
        int red = 0;
        int green = 255;
        int blue = 0;
        int alpha = 255;
        const std::string &text = labelsColor[static_cast<size_t>(classId)];
        if (std::sscanf(text.c_str(), "rgba(%d, %d, %d, %d)", &red, &green, &blue, &alpha) >= 3) {
            return cv::Scalar(blue, green, red);
        }
    }

    return cv::Scalar(0, 255, 0);
}

std::string makeLabel(const DetectResultBox &box)
{
    std::ostringstream label;
    label << (box.label.empty() ? std::to_string(box.classId) : box.label)
          << ' ' << std::fixed << std::setprecision(2) << box.confidence;
    return label.str();
}
}

void draw(cv::Mat &mat, std::vector<DetectResultBox> &detectResultBoxes)
{
    if (mat.empty()) {
        return;
    }

    for (const DetectResultBox &box : detectResultBoxes) {
        const int x = std::max(0, static_cast<int>(std::round(box.x)));
        const int y = std::max(0, static_cast<int>(std::round(box.y)));
        const int width = std::min(mat.cols - x, std::max(1, static_cast<int>(std::round(box.width))));
        const int height = std::min(mat.rows - y, std::max(1, static_cast<int>(std::round(box.height))));
        if (width <= 0 || height <= 0) {
            continue;
        }

        const cv::Scalar color = colorForClass(box.classId);
        const cv::Rect rect(x, y, width, height);
        cv::rectangle(mat, rect, color, 2, cv::LINE_AA);

        const std::string label = makeLabel(box);
        int baseline = 0;
        const cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int labelTop = std::max(0, y - labelSize.height - baseline - 4);
        const int labelWidth = std::min(labelSize.width + 6, mat.cols - x);
        const int labelHeight = labelSize.height + baseline + 4;
        if (labelWidth > 0 && labelHeight > 0) {
            cv::rectangle(mat, cv::Rect(x, labelTop, labelWidth, labelHeight), color, cv::FILLED);
            cv::putText(
                mat,
                label,
                cv::Point(x + 3, labelTop + labelSize.height + 1),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(0, 0, 0),
                1,
                cv::LINE_AA);
        }
    }
}
