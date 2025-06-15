# tensor-engine

Demo engine like `tensorrt` or `onnxruntime` to learn C++

```
- include/   # 头文件
- src/       # 源文件
    - parser/  # 模型解析器（如ONNXParser）
    - graph/  # 图结构和图优化
    - kernels/ # 核函数（CUDA代码）
    - memory/ # 内存管理
    - execution/ # 执行引擎
    - test/      # 测试代码

```

- 线程池并发执行计算图（拓扑顺序）
- shared ptr 管理tensor分配的内存
- 算子融合（element wise）
  - mm + add
- 图优化
  - remove dead node
  - const fold
- 并发安全 hashmap（左倾红黑树）、queue
- 类似 ATen 的 AT_DISPATCH_FLOATING_TYPES_AND_HALF
- cuda stream 串联 malloc、memcpy、算子计算
- Graph(weight、结构) 和 input、计算中间结果 分离
- 同时支持CUDA、CPU
- ONNX 算子实现规约
- ONNX 矩阵乘法广播
- CUDA operator & fp16 特化实现
  - tiled mm
  - add
  - relu


- initializer_list
- std::function
- shared ptr
- unique ptr