#!/bin/bash

# --- 1. 基本变量可配置 ---
IMAGE_NAME="cuda-cmake-demo"
DOCKERFILE_PATH="build/Dockerfile"
CONTEXT_DIR="."
PLATFORM="linux/amd64"

# --- 2. 可选参数解析（手工传递镜像名等） ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--tag)
      IMAGE_NAME="$2"
      shift 2
      ;;
    -p|--platform)
      PLATFORM="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [-t image_name] [-p platform]"
      exit 1
      ;;
  esac
done

# --- 3. 构建镜像 ---
echo "Building Docker image: $IMAGE_NAME"
docker buildx build --platform "$PLATFORM" -t "$IMAGE_NAME" --load -f "$DOCKERFILE_PATH" "$CONTEXT_DIR"

# --- 4. 构建完成提醒 ---
echo "Docker image '$IMAGE_NAME' built successfully!"