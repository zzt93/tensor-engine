#include "../../include/graph.h"

#include <unordered_set>

#include "queue"
#include "../../include/operator.h"

using namespace tensorengine;
using namespace std;

void GraphOptimizer::fuseLayers(Graph& g) {
    auto outputOfNode = unordered_map<string, shared_ptr<Node>>{};
    auto queue = std::deque<shared_ptr<Node>>{};
    auto inputSet = unordered_set<string>{};
    for (auto input : g.inputs) {
        inputSet.emplace(input.name);
    }

    for (auto node : g.nodes) {
        for (auto output : node->outputs) {
            outputOfNode[output] = node;
        }
    }

    for (auto tensor_meta : g.outputs) {
        auto n = outputOfNode.find(tensor_meta.name);
        assert(n != outputOfNode.end());
        queue.push_back(n->second);
    }
    while (!queue.empty()) {
        auto node = queue.front();
        queue.pop_front();

        set<string> pres;
        for (auto output : node->inputs_) {
            if (outputOfNode.contains(output)) {
                auto pre = outputOfNode[output];
                queue.push_back(pre);
                pres.emplace(pre->op_type);
            } else {
                assert(inputSet.contains(output));
            }
        }
        // pattern match
        auto p = OperatorPattern{pres, node->name};
        if (g_fusePattern.contains(p)) {
            auto fuse_op = g_fusePattern[p];
            vector<string> fuse_inputs{};
            vector<Attribute> fuse_attrs{};
            g.nodes.remove(node);
            string fuse_name = node->name;
            for (auto output : node->inputs_) {
                if (outputOfNode.contains(output)) {
                    auto pre = outputOfNode[output];
                    fuse_name += "_" + pre->name;
                    fuse_inputs.insert(fuse_inputs.end(), pre->inputs_.cbegin(), pre->inputs_.cend());
                    g.nodes.remove(pre);
                }
            }
            // 因为合并消除的都是中间变量，不会存储，不需要单独删除
            auto fuse_node = make_shared<Node>(fuse_name, fuse_op, fuse_inputs, node->outputs, fuse_attrs);
            g.nodes.push_back(fuse_node);
        }
    }

}

void GraphOptimizer::removeDeadNodes(Graph& g) {
    auto seen = unordered_set<shared_ptr<Node>>{};
    auto outputMap = unordered_map<string, shared_ptr<Node>>{};
    auto queue = std::deque<shared_ptr<Node>>{};
    auto inputSet = unordered_set<string>{};
    for (auto input : g.inputs) {
        inputSet.emplace(input.name);
    }

    for (auto node : g.nodes) {
        for (auto output : node->outputs) {
            outputMap[output] = node;
        }
    }
    for (auto tensor_meta : g.outputs) {
        auto n = outputMap.find(tensor_meta.name);
        assert(n != outputMap.end());
        queue.push_back(n->second);
    }
    while (!queue.empty()) {
        auto node = queue.front();
        queue.pop_front();
        seen.emplace(node);

        for (auto output : node->inputs_) {
            if (outputMap.contains(output)) {
                queue.push_back(outputMap[output]);
            } else {
                logger.error("unexpected node output: " + output);
                assert(inputSet.contains(output));
            }
        }
    }

    // remove dead node: nodes
    set<string> weight, inputs, outputs;
    for (auto it = g.nodes.begin(); it != g.nodes.end();) {
        if (!seen.contains(*it)) {
            it = g.nodes.erase(it);
        } else {
            ++it;
            for (auto input : (*it)->inputs_) {
                weight.emplace(input);
            }
        }
    }
    // delete weights & inputs & outputs if only those node use

}

void GraphOptimizer::constFolding(Graph &g) {
    // deduce dim
    unordered_map<string, TensorMeta> input_meta;
    for (auto tensor_meta : g.inputs) {
        input_meta.emplace(tensor_meta.name, tensor_meta);
    }
    unordered_map<string, vector<shared_ptr<Node>>> node_input;
    unordered_map<shared_ptr<Node>, int> node_extra;

    for (auto node : g.nodes) {
        for (auto input : node->inputs_) {
            if (!node_input.contains(input)) {
                node_input[input] = vector{node};
            } else {
                node_input[input].push_back(node);
            }
        }
        node_extra[node] = 0;
    }

    // mimic exec
    queue<string> q;
    for (auto p : g.weights) {
        q.push(p.first);
    }
    while (!q.empty()) {
        auto const_input = q.front();
        q.pop();

        for (auto node : node_input[const_input]) {
            node_extra[node] += 1;
            if (node_extra[node] == node->inputs_.size()) {
                // exec
                vector<shared_ptr<Tensor>> input(node->inputs_.size());
                for (int i = 0; i < node->inputs_.size(); ++i) {
                    input[i] = g.weights[node->inputs_[i]];
                }
                vector<shared_ptr<Tensor>> output{};
                OperatorContext ctx(node->attributes);
                auto f = M_OP_MAP[node->op_type];
                f(input, output, ctx);
                assert(output.size() == node->outputs.size());
                auto data_type = input[0]->data_type();
                auto output_dims = node->calOutputDim(input_meta);
                assert(output_dims.size() == node->outputs.size());
                // add new weight & input
                for (int i = 0; i < node->outputs.size(); ++i) {
                    q.push(node->outputs[i]);

                    g.weights[node->outputs[i]] = output[i];
                    auto meta = TensorMeta{node->outputs[i], output_dims[i], data_type};
                    g.inputs.emplace(meta);
                    input_meta.emplace(meta.name, meta);
                }
                // remove nodes & weights & inputs
                g.nodes.remove(node);
                for (auto input_name : node->inputs_) {
                    g.weights.erase(input_name);
                    g.inputs.erase(TensorMeta{input_name, {}, data_type});
                }
            }
        }
    }
}

std::unique_ptr<ExecutionGraph> Graph::parse() {
    vector<shared_ptr<ParsedNode>> parsedNodes(nodes.size());
    for (const auto &n: nodes) {
        parsedNodes.push_back(make_shared<ParsedNode>(n->inputs_, n->outputs, n->op_type, n->attributes));
    }
    vector<pair<int, int>> edges;

    // 一个变量只能被一个节点 “生产”，可以被多个节点 “消费”
    unordered_map<string, shared_ptr<ParsedNode>> allOutputName;
    unordered_map<shared_ptr<ParsedNode>, int> nodeNo;
    unordered_map<int, shared_ptr<ParsedNode>> idxNode;

    int i = 0;
    for (const auto &n: parsedNodes) {
        nodeNo[n] = i;
        idxNode[i] = n;
        i++;

        for (const auto &item: n->outputs) {
            if (allOutputName.contains(item)) {
                throw std::runtime_error("Invalid graph: multiple node has same output name: " + item);
            } else {
                allOutputName[item] = n;
            }
        }
    }
    for (const auto &n: parsedNodes) {
        for (const auto &in: n->inputs) {
            if (allOutputName.find(in) != allOutputName.end()) {
                const auto &out = allOutputName[in];
                edges.push_back(make_pair(nodeNo[out], nodeNo[n]));
                out->addTo(n);
            }
        }
    }

    vector<int> indegree(nodes.size(), 0);     // 入度数组
    for (const auto &e: edges) {
        indegree[e.second]++;
    }
    vector<std::shared_ptr<ParsedNode>> start;
    for (int j = 0; j < indegree.size(); ++j) {
        if (indegree[i] == 0) {
            start.push_back(idxNode[i]);
        }
    }

    return make_unique<ExecutionGraph>(start, weights, inputs, outputs);
}

std::vector<std::vector<int>> Node::calOutputDim(unordered_map<string, TensorMeta>& metas) {
    auto outputDim = vector<vector<int>>(outputs.size(), vector<int>());
    auto operator_desc = M_OP_DESC[op_type];
    vector<TensorMeta> inputMeta;
    for (auto input : inputs_) {
        auto it = metas.find(input);
        inputMeta.push_back(it->second);
    }
    switch (operator_desc.broadcast) {
        case BroadCastType::None:
            break;
        case BroadCastType::Unidirectional:
            break;
        case BroadCastType::Multidirectional:
            break;

    }
    return outputDim;
}

void Graph::opt() {
    GraphOptimizer optimizer;
    optimizer.removeDeadNodes(*this);
    optimizer.fuseLayers(*this);
    optimizer.constFolding(*this);
}

namespace tensorengine {
    std::unordered_map<OperatorPattern, std::string, OperatorPatternHash> g_fusePattern = {
        {OperatorPattern{set<string>{OP_GEMM}, OP_ADD}, F_OP_MMA},
        {OperatorPattern{set<string>{OP_EXPAND}, OP_ADD}, F_OP_BATCH_ADD},
        {OperatorPattern{set<string>{OP_EXPAND}, OP_GEMM}, F_OP_BATCH_MM},
    };
}
