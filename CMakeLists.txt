cmake_minimum_required(VERSION 3.21)
project(tensor_engine LANGUAGES CXX)

# 尝试启用CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    add_definitions(-DUSE_CUDA)
    enable_language(CUDA)
    message(STATUS "CUDA detected: ${CMAKE_CUDA_COMPILER}")
else()
    message(STATUS "CUDA NOT FOUND, building without CUDA support.")
endif()

set(CMAKE_CXX_STANDARD 20)

#file(GLOB_RECURSE HEADER_LIST "${PROJECT_SOURCE_DIR}/include/*.h")
#
#set(HEADER_TEST_TARGETS "")
#
#foreach(header_file ${HEADER_LIST})
#    get_filename_component(header_name ${header_file} NAME_WE)
#    set(test_src "${CMAKE_BINARY_DIR}/test_header_${header_name}.cpp")
#    file(WRITE ${test_src} "#include \"${header_file}\"\n")
#    add_library(test_header_${header_name} STATIC ${test_src})
#    list(APPEND HEADER_TEST_TARGETS test_header_${header_name})
#endforeach()

# 定义一个伪目标，执行这句可以检测所有头文件
#add_custom_target(check_headers ALL DEPENDS ${HEADER_TEST_TARGETS})
#add_compile_options(-Wall -Wextra  -pedantic)

add_executable(tensor_engine main.cpp
        include/parser.h
        include/graph.h
        include/memory.h
        include/engine.h
        include/tensor.h
        include/device.h
        include/type.h
        src/parser/onnx_parser.cpp
        include/config.h
        src/test/config.cpp
        include/operator.h
        include/enum.h
        src/execution/engine.cpp
        include/util.h
        src/util/util.cpp
        src/graph/graph.cpp
        src/device/device.cpp
        src/util/logger.cpp
        src/kernels/operator.cpp
        src/kernels/gpu/add_fp16.cu
        include/kernel/cuda/mm.cu
        include/kernel/cuda/relu.cu
        include/kernel/cpu/simple_operator.tpp
        include/kernel/cuda/add.cu
        src/kernels/gpu/mm_fp16.cu
        src/kernels/gpu/relu_fp16.cu
        src/graph/tensor.cpp
        src/device/CUDADevice.cu)
