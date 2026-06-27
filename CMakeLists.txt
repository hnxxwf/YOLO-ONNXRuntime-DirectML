cmake_minimum_required(VERSION 3.15)

project(YOLO_ONNXRuntime_DirectML
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_BUILD_TYPE "Release")

#set the OpenCV static library path
set(OpenCV_Static_Root_Path
    "${CMAKE_CURRENT_LIST_DIR}/third_party/OpenCV-Static-4.14.0"
    CACHE PATH "OpenCV static install prefix"
)
#set the ONNXRuntime static library path
set(ONNXRuntime_Static_Root_Path
    "${CMAKE_CURRENT_LIST_DIR}/third_party/ONNXRuntime-Static-1.28.0"
    CACHE PATH "ONNX Runtime static install prefix"
)
#set the YoloOrtDml SDK install path!!!!!!!!!!!!!!!!!
set(YoloOrtDml_Shared_Install_Path
    "${CMAKE_CURRENT_LIST_DIR}/install"
    CACHE PATH "YoloOrtDml SDK install prefix"
)
set(YoloOrtDml_DirectML_DLL
    "${ONNXRuntime_Static_Root_Path}/bin/DirectML.dll"
    CACHE FILEPATH "DirectML runtime DLL"
)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
set(CMAKE_INSTALL_PREFIX "${YoloOrtDml_Shared_Install_Path}" CACHE PATH "Install prefix" FORCE)

add_subdirectory(src)
add_subdirectory(example)
