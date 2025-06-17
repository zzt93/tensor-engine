#include <iostream>

#include "include/config.h"
#include "include/parser.h"
#include "include/graph.h"
#include "include/engine.h"

using namespace tensorengine;
using namespace std;

shared_ptr<Tensor> getInputData();

int main() {
    // 解析模型
    OnnxParser parser;
    Graph graph;
    parser.parse("model.onnx", graph);

    // 初始化引擎
    auto engine = make_shared<InferenceEngine>(graph); // 分配内存/加载权重

    // 执行推理
    auto context = engine->createExecutionContext();
    shared_ptr<Tensor> input = getInputData();
    if (g_testing) cout << "input: " << *input << endl;
    context->setInput("input_0", input);
    bool exec = context->execute();
    if (exec) {
        shared_ptr<Tensor> output = context->getOutput("output_0");
        cout << "output: " << *output << endl;
    }
}

shared_ptr<Tensor> getInputData() {
    auto input = std::make_shared<Tensor>(g_dims[0], DataType::FP32, g_device);
//    auto input = std::make_shared<Tensor>(std::vector<int>{1, 15, 256}, DataType::FP32, g_device);
    input->fill(rands(input->size(), 0, 1));
    return input;
}
