# tensor-engine

Demo engine like `tensorrt` or `onnxruntime` to learn C++

```
- include/   # 头文件
- src/       # 源文件
    - parser/  # 模型解析器（mock ONNX）
    - graph/  # 图结构和图优化
    - kernels/ # 核函数（CUDA代码）
    - device/ # device管理
    - execution/ # 执行引擎
    - test/      # 测试代码

```

- 算子融合（element wise）
  - mm + add
  - expand + add
  - expand + mm
- 图优化
  - remove dead node
  - const fold
- cuda stream 串联 malloc、memcpy、算子计算
- Graph(weight、结构) 和 input、计算中间结果 分离
- 同时支持CUDA、CPU
  - CPU环境 线程池并发执行计算图（拓扑顺序）
  - CUDA环境 stream异步统一提交kernel（CUDA Graph可能更加并行）
- ONNX 
  - 算子实现规约
  - IR 约定：算子没有直接指针，通过input、output关联；weight在graph，方便复用
- ONNX 广播
  - 矩阵乘法batch维度广播（gpu、cpu）
    - 广播输入B
    - batch小于8直接使用shared memory
  - 加法batch维度广播（gpu、cpu）
  - 加法双向广播（cpu）
- CUDA operator & fp16 特化实现
  - tiled mm(NT format)
  - add
  - relu

---
c++

- 可变数量参数 & initializer_list
- std::function
- std::future
- shared ptr 管理tensor分配的内存
  - dynamic cast & dynamic_pointer_cast
- unique ptr
- 单例：函数静态局部变量
- 并发安全 hashmap（左倾红黑树）、blocking queue
- lock_guard & unique_lock & condition
  - `cond.wait(lock, [this] { return this->finished(); });`
- iterator_base & const & non const
- 类似 ATen 的宏（AT_DISPATCH_FLOATING_TYPES_AND_HALF），动态入参初始化模版参数
  - `if constexpr (std::is_same_v<T, __half>) `
