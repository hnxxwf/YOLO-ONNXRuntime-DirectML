cmake_minimum_required(VERSION 3.15)

project(YoloOrtDml
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_INSTALL_PREFIX "${YoloOrtDml_Shared_Install_Path}" CACHE PATH "Install prefix" FORCE)
 
list(APPEND CMAKE_PREFIX_PATH
    "${OpenCV_Static_Root_Path}"
    "${ONNXRuntime_Static_Root_Path}"
)

find_package(OpenCV CONFIG REQUIRED COMPONENTS world)
find_package(ONNXRuntime CONFIG REQUIRED)
include(CMakePackageConfigHelpers)

add_library(YoloOrtDmlLib SHARED)
add_library(YoloOrtDml::YoloOrtDml ALIAS YoloOrtDmlLib)

target_sources(YoloOrtDmlLib
    PRIVATE
        src/YoloOrtDml.cpp
        src/draw.cpp
        src/infer.cpp
        src/postprocess.cpp
        src/preprocess.cpp
)

target_compile_features(YoloOrtDmlLib
    PUBLIC
        cxx_std_17
)

target_include_directories(YoloOrtDmlLib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_definitions(YoloOrtDmlLib
    PRIVATE
        YOLOORTDML_BUILD_SHARED
    INTERFACE
        YOLOORTDML_USE_SHARED
)

target_link_libraries(YoloOrtDmlLib
    PUBLIC
        opencv_world
    PRIVATE
        onnxruntime::onnxruntime
)

set_target_properties(YoloOrtDmlLib PROPERTIES
    OUTPUT_NAME YoloOrtDml
    EXPORT_NAME YoloOrtDml
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

add_custom_command(TARGET YoloOrtDmlLib POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${YoloOrtDml_DirectML_DLL}"
        "$<TARGET_FILE_DIR:YoloOrtDmlLib>/DirectML.dll"
    VERBATIM
)

install(TARGETS YoloOrtDmlLib
    EXPORT YoloOrtDmlTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    include/YoloOrtDml.h
    include/YoloOrtDmlExport.h
    include/model.h
    DESTINATION include
)

install(DIRECTORY "${OpenCV_Static_Root_Path}/include/"
    DESTINATION include
)

install(DIRECTORY "${OpenCV_Static_Root_Path}/lib/"
    DESTINATION lib
)

install(DIRECTORY "${OpenCV_Static_Root_Path}/bin/"
    DESTINATION bin
)

install(FILES "${YoloOrtDml_DirectML_DLL}" DESTINATION bin)

install(EXPORT YoloOrtDmlTargets
    FILE YoloOrtDmlTargets.cmake
    NAMESPACE YoloOrtDml::
    DESTINATION lib/cmake/YoloOrtDml
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/YoloOrtDmlConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/YoloOrtDmlConfig.cmake"
    INSTALL_DESTINATION lib/cmake/YoloOrtDml
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/YoloOrtDmlConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/YoloOrtDmlConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/YoloOrtDmlConfigVersion.cmake"
    DESTINATION lib/cmake/YoloOrtDml
)
