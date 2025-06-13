# tensor-engine

Demo engine like `tensorrt` or `onnxruntime` to learn C++

```
- include/   # 头文件
- src/       # 源文件
    - Parser/  # 模型解析器（如ONNXParser）
    - Graph/  # 图结构和图优化
    - Layers/ # 各种层的实现（如ConvLayer, ReLULayer等）
    - Kernels/ # 核函数（CUDA代码）
    - Memory/ # 内存管理
    - Execution/ # 执行引擎
    - test/      # 测试代码

```

- 线程池并发执行计算图（拓扑顺序）
- 并发安全map、queue
- 类似 ATen 的 AT_DISPATCH_FLOATING_TYPES_AND_HALF
- fp16 特化实现
- cuda stream 串联算子计算