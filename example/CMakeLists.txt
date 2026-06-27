cmake_minimum_required(VERSION 3.15)

project(YOLOOrtDmlExample LANGUAGES CXX)

add_executable(YOLOOrtDmlExample )
target_sources(YOLOOrtDmlExample
    PRIVATE
        main.cpp
)

target_compile_features(YOLOOrtDmlExample
    PRIVATE
        cxx_std_17
)

target_link_libraries(YOLOOrtDmlExample
    PRIVATE
        YoloOrtDml::YoloOrtDml
)

add_custom_command(TARGET YOLOOrtDmlExample POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:YoloOrtDmlLib>"
        "${YoloOrtDml_DirectML_DLL}"
        $<TARGET_FILE_DIR:YOLOOrtDmlExample>
    VERBATIM
)
