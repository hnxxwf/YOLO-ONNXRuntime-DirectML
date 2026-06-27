#ifndef YOLOORTDML_H
#define YOLOORTDML_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

#include "model.h"
#include "YoloOrtDmlExport.h"

class YOLOORTDML_API YoloOrtDml
{
public:
    YoloOrtDml();
    ~YoloOrtDml();
    
    void setModel(std::string& modelPath);              //set model path
    void setConfThreshold(float confThreshold);         //set confidence threshold
    void setNmsThreshold(float nmsThreshold);           //set nms threshold
    
    void preprocess(cv::Mat &mat);                      //preprocess image
    void infer();                                       //infer
    std::vector<DetectResultBox> postprocess();         //finishing confidence threshold and nms threshold filter,return resultBoxes
    cv::Mat draw();                                     //(optional)draw result,you could use or not use this function by yourself

private:
    struct Pimpl;
    std::unique_ptr<Pimpl> pimpl;
};

#endif // YOLOORTDML_H
