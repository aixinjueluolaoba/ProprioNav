cmake_minimum_required(VERSION 3.22.1)
project(proprionav_jni LANGUAGES CXX)

set(PROPRIONAV_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../..")

add_library(ncnn_rust SHARED IMPORTED)
set_target_properties(
    ncnn_rust
    PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/../jniLibs/${ANDROID_ABI}/libncnn_rust.so"
    IMPORTED_NO_SONAME TRUE
)

add_library(proprionav_jni SHARED proprionav_jni.cpp)
target_include_directories(
    proprionav_jni
    PRIVATE "${PROPRIONAV_ROOT}/ncnn_rust/include"
)
target_link_libraries(proprionav_jni PRIVATE ncnn_rust log)
