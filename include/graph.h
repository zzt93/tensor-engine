#pragma once


#include "set"

#include "iostream"
#include "tensor.h"
#include "util.h"

namespace tensorengine {


    class Attribute {
    public:
        std::string name;
        std::variant<int, float, std::string, Tensor*> value;
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

    };

    class ParsedGraph {
        bool isReady;
    public:
        std::vector<std::shared_ptr<ParsedNode>> start;
        std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors;

        ParsedGraph(std::vector<std::shared_ptr<ParsedNode>> start, std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors): start(std::move(start)), tensors(std::move(tensors)) {
        }

        void opt();
        void setInput(const std::string& input, std::shared_ptr<Tensor> t);
    };

    class Node {
    public:
        std::string name;
        std::string op_type;  // "Conv", "Relu", etc.
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
        std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors;
        std::set<std::string> inputs;
        std::set<std::string> outputs;

        void addNode(std::shared_ptr<Node> node) {
            nodes.push_back(std::move(node));
        }

        void addWeight(const std::string& input, std::shared_ptr<Tensor> t) {
            inputs.emplace(input);
            tensors[input] = std::move(t);
        }

        void setInput(const std::string& input, std::shared_ptr<Tensor> t) {
            addWeight(input, std::move(t));
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

