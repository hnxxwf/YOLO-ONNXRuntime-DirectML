#ifndef MODEL_H
#define MODEL_H
#include <string>
#include <vector>
struct DetectResultBox
{
    float x;
    float y;
    float width;
    float height;
    float confidence;
    int classId;
    std::string label;
};

//class names
inline  const std::vector<std::string> labelsName=
{
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack", "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball", "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza","donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse",  "remote",  "keyboard",  "cell phone",  "microwave",  "oven",  "toaster",  "sink",
    "refrigerator",  "book",  "clock",  "vase",  "scissors",  "teddy bear","hair drier",  "toothbrush"
};

//draw box color,rgba
inline  const std::vector<std::string> labelsColor=
{
    "rgba(230, 25, 75, 255)", "rgba(60, 180, 75, 255)", "rgba(255, 225, 25, 255)", "rgba(0, 130, 200, 255)",
    "rgba(245, 130, 48, 255)", "rgba(145, 30, 180, 255)", "rgba(70, 240, 240, 255)", "rgba(240, 50, 230, 255)",
    "rgba(210, 245, 60, 255)", "rgba(250, 190, 190, 255)", "rgba(0, 128, 128, 255)", "rgba(230, 190, 255, 255)",
    "rgba(170, 110, 40, 255)", "rgba(255, 250, 200, 255)", "rgba(128, 0, 0, 255)", "rgba(0, 0, 255, 255)",
    "rgba(128, 128, 0, 255)", "rgba(255, 215, 180, 255)", "rgba(0, 0, 128, 255)", "rgba(128, 128, 128, 255)",
    "rgba(255, 99, 132, 255)", "rgba(54, 162, 235, 255)", "rgba(255, 206, 86, 255)", "rgba(75, 192, 192, 255)",
    "rgba(153, 102, 255, 255)", "rgba(255, 159, 64, 255)", "rgba(46, 204, 113, 255)", "rgba(231, 76, 60, 255)",
    "rgba(52, 152, 219, 255)", "rgba(155, 89, 182, 255)", "rgba(241, 196, 15, 255)", "rgba(26, 188, 156, 255)",
    "rgba(230, 126, 34, 255)", "rgba(149, 165, 166, 255)", "rgba(52, 73, 94, 255)", "rgba(192, 57, 43, 255)",
    "rgba(41, 128, 185, 255)", "rgba(142, 68, 173, 255)", "rgba(39, 174, 96, 255)", "rgba(243, 156, 18, 255)",
    "rgba(22, 160, 133, 255)", "rgba(211, 84, 0, 255)", "rgba(127, 140, 141, 255)", "rgba(44, 62, 80, 255)",
    "rgba(255, 87, 51, 255)", "rgba(199, 0, 57, 255)", "rgba(144, 12, 63, 255)", "rgba(88, 24, 69, 255)",
    "rgba(0, 168, 255, 255)", "rgba(156, 136, 255, 255)", "rgba(251, 197, 49, 255)", "rgba(76, 209, 55, 255)",
    "rgba(232, 65, 24, 255)", "rgba(0, 151, 230, 255)", "rgba(140, 122, 230, 255)", "rgba(225, 177, 44, 255)",
    "rgba(68, 189, 50, 255)", "rgba(194, 54, 22, 255)", "rgba(39, 60, 117, 255)", "rgba(53, 59, 72, 255)",
    "rgba(112, 111, 211, 255)", "rgba(64, 115, 158, 255)", "rgba(72, 126, 176, 255)", "rgba(127, 143, 166, 255)",
    "rgba(255, 107, 107, 255)", "rgba(72, 219, 251, 255)", "rgba(29, 209, 161, 255)", "rgba(254, 202, 87, 255)",
    "rgba(84, 160, 255, 255)", "rgba(95, 39, 205, 255)", "rgba(16, 172, 132, 255)", "rgba(255, 159, 243, 255)",
    "rgba(243, 104, 224, 255)", "rgba(0, 210, 211, 255)", "rgba(87, 101, 116, 255)", "rgba(34, 47, 62, 255)",
    "rgba(255, 195, 18, 255)", "rgba(196, 229, 56, 255)", "rgba(18, 203, 196, 255)", "rgba(237, 76, 103, 255)"
};
#endif // MODEL_H
