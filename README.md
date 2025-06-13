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
- 并发安全 hashmap（红黑树）、queue
- 类似 ATen 的 AT_DISPATCH_FLOATING_TYPES_AND_HALF
- fp16 特化实现
- cuda stream 串联 malloc、memcpy、算子计算
- Graph(weight、结构) 和 input、计算中间结果 分离
- 同时支持CUDA、CPU
- CUDA operator
  - tiled mm
  - add
  - relu