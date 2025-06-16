# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 安装构建依赖
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ninja-build \
        && rm -rf /var/lib/apt/lists/*

# 为非交互式应用设置默认工作目录
WORKDIR /workspace

# 拷贝所有源代码
COPY . .


# 默认命令（直接编译 Release，可改为 Debug）
CMD ["bash"]