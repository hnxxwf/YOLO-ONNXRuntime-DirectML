#ifndef DRAW_H
#define DRAW_H
#include <opencv2/opencv.hpp>
#include "model.h"
void draw(cv::Mat& mat, std::vector<DetectResultBox>& detectResultBoxes);
#endif // DRAW_H