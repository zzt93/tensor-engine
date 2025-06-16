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

        bool isEnd() const {
            return to.empty();
        }
    };

    class ExecutionGraph {
        std::vector<std::shared_ptr<ParsedNode>> start_;
        std::unordered_map<std::string, std::shared_ptr<Tensor>> weights_;
        std::set<TensorMeta> configInput_;
        std::set<TensorMeta> configOutput_;
    public:
        ExecutionGraph(std::vector<std::shared_ptr<ParsedNode>> start, std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors, std::set<TensorMeta> configInput, std::set<TensorMeta> configOutput): start_(std::move(start)), weights_(std::move(tensors)), configInput_(std::move(configInput)), configOutput_(configOutput) {
        }
        const std::set<TensorMeta>& getConfigInput() {
            return configInput_;
        }
        const std::set<TensorMeta>& getConfigOutput() {
            return configOutput_;
        }
        const std::vector<std::shared_ptr<ParsedNode>>& getStartNode() const {
            return start_;
        }
        const std::unordered_map<std::string, std::shared_ptr<Tensor>>& getWeight() const {
            return weights_;
        }
    };

    class Node {
    public:
        std::string name;
        std::string op_type;  // "Conv", "Relu", etc.
        // ONNX 标准规定：某算子第 N 个参数，就取 inputs_ 列表里的第 N 个名字
        std::vector<std::string> inputs_;
        std::vector<std::string> outputs;
        // 层参数 (如卷积的 kernel_size)
        std::vector<Attribute> attributes;

        Node(std::string name, const std::string &opType, const std::vector<std::string> &inputs, const std::vector<std::string> &outputs, const std::vector<Attribute> &attributes) : name(std::move(
                name)), op_type(opType), inputs_(inputs), outputs(outputs), attributes(attributes) {}

        std::vector<std::vector<int>> calOutputDim(std::unordered_map<std::string, TensorMeta>& metas);
    };

    class Graph {
    private:
        std::list<std::shared_ptr<Node>> nodes;
        std::unordered_map<std::string, std::shared_ptr<Tensor>> weights;
        std::set<TensorMeta> inputs;
        std::set<TensorMeta> outputs;
    public:
        friend class GraphOptimizer;


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

        void opt();
        std::unique_ptr<ExecutionGraph> parse();

        ~Graph() {
        }

    };


    class GraphOptimizer {
        inline static Logger logger{};

    public:
        // output 反向推导 input，如果模式符合，融合替换节点
        static void fuseLayers(Graph&);
        // output 反向推导 input，删除未使用节点
        static void removeDeadNodes(Graph&);
        // input 前向推导，如果input都是常量，计算并删除节点
        static void constFolding(Graph&);
    };

    class OperatorPattern {
    public:
        std::set<std::string> inputOps;
        std::string outputOp;

        OperatorPattern(const std::set<std::string> &input_ops, const std::string &output_op)
            : inputOps(input_ops),
              outputOp(output_op) {
        }

        friend bool operator==(const OperatorPattern &lhs, const OperatorPattern &rhs) {
            return lhs.inputOps == rhs.inputOps
                   && lhs.outputOp == rhs.outputOp;
        }

        friend bool operator!=(const OperatorPattern &lhs, const OperatorPattern &rhs) {
            return !(lhs == rhs);
        }

        friend std::size_t hash_value(const OperatorPattern &obj) {
            std::size_t seed = 0x495FD938;
            seed ^= (seed << 6) + (seed >> 2) + 0x6F246971 + setHash(obj.inputOps);
            seed ^= (seed << 6) + (seed >> 2) + 0x0CC0D21B + std::hash<std::string>()(obj.outputOp);
            return seed;
        }
    };

    struct OperatorPatternHash {
        size_t operator()(const OperatorPattern& k) const {
            return hash_value(k);
        }
    };


    extern std::unordered_map<OperatorPattern, std::string, OperatorPatternHash> g_fusePattern;

}

