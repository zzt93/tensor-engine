cmake_minimum_required(VERSION 3.27)
project(tensor_engine)

set(CMAKE_CXX_STANDARD 20)

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
        src/util/logger.cpp)
