docker run -it --rm -v "$(pwd)":/workspace cuda-cmake-demo /bin/bash

cmake -S . -B nvcc-cmake-build-debug  # -DCMAKE_BUILD_TYPE=Debug 也行
cmake --build nvcc-cmake-build-debug