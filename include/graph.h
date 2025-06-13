#pragma once


#include "set"

#include "iostream"
#include "tensor.h"
#include "util.h"

namespace tensorengine {

    class TensorMeta {
    public:
        const std::string name;
        const std::vector<int> dim;
        const DataType type;

        TensorMeta(std::string name, std::vector<int> dim, DataType type): name(name), dim(std::move(dim)), type(type){}

        bool operator<(const TensorMeta &rhs) const {
            return name < rhs.name;
        }

        friend std::ostream &operator<<(std::ostream &os, const TensorMeta &meta) {
            os << "name: " << meta.name << " dim: " << tostring(meta.dim) << " type: " << tostring(meta.type);
            return os;
        }
    };

    class ParsedNode {
    public:
//        std::vector<std::weak_ptr<ParsedNode>> from;
        std::vector<std::shared_ptr<ParsedNode>> to;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        std::string op_type;  // "Conv", "Relu", etc.
        std::vector<Attribute> attributes;


        ParsedNode(const std::vector<std::string> &inputs, const std::vector<std::string> &outputs,
                               const std::string &opType, const std::vector<Attribute> &attributes) : inputs(inputs), outputs(outputs), op_type(opType), attributes(attributes) {}

        void addTo(std::shared_ptr<ParsedNode> t) {
            to.push_back(t);
        }

        bool isEnd() {
            return to.empty();
        }
    };

    class ParsedGraph {
        std::vector<std::shared_ptr<ParsedNode>> start;
        std::unordered_map<std::string, std::shared_ptr<Tensor>> weights;
        std::set<TensorMeta> configInput;
        std::set<TensorMeta> configOutput_;
    public:
        ParsedGraph(std::vector<std::shared_ptr<ParsedNode>> start, std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors, std::set<TensorMeta> configInput, std::set<TensorMeta> configOutput): start(std::move(start)), weights(std::move(tensors)), configInput(std::move(configInput)), configOutput_(configOutput) {
        }
        void opt();
        const std::set<TensorMeta>& getConfigInput() {
            return configInput;
        }
        const std::set<TensorMeta>& getConfigOutput() {
            return configOutput_;
        }
        const std::vector<std::shared_ptr<ParsedNode>>& getStartNode() const {
            return start;
        }
    };

    class Node {
    public:
        std::string name;
        std::string op_type;  // "Conv", "Relu", etc.
        // ONNX 标准规定：某算子第 N 个参数，就取 inputs_ 列表里的第 N 个名字
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        // 层参数 (如卷积的 kernel_size)
        std::vector<Attribute> attributes;

        Node(std::string name, const std::string &opType, const std::vector<std::string> &inputs, const std::vector<std::string> &outputs, const std::vector<Attribute> &attributes) : name(std::move(
                name)), op_type(opType), inputs(inputs), outputs(outputs), attributes(attributes) {}
    };

    class Graph {
    private:
    public:
        std::vector<std::shared_ptr<Node>> nodes;
        std::unordered_map<std::string, std::shared_ptr<Tensor>> weights;
        std::set<TensorMeta> inputs;
        std::set<TensorMeta> outputs;

        void addNode(std::shared_ptr<Node> node) {
            nodes.push_back(std::move(node));
        }

        void addWeight(const std::string& input, std::shared_ptr<Tensor> t) {
            weights[input] = std::move(t);
        }

        void addInput(const TensorMeta& input) {
            inputs.emplace(input);
        }

        void chooseOutput(const TensorMeta& output) {
            outputs.emplace(output);
        }

        std::unique_ptr<ParsedGraph> parse();

        ~Graph() {
        }
    };


    class GraphOptimizer {
    public:
        void fuseLayers(ParsedGraph*);
        void removeDeadNodes(ParsedGraph*);
    };

}

