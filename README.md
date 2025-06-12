# tensor-engine

Demo engine to learn `tensorrt`

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