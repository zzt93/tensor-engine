#include "../../include/graph.h"

#include <unordered_set>
#include <sstream>
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
            if (outputOfNode.find(output) != outputOfNode.end()) {
                auto pre = outputOfNode[output];
                pres.emplace(pre->op_type);
            } else {
                ASSERT_MSG(inputSet.find(output) != inputSet.end() || g.weights.find(output) != g.weights.end(), "unexpected node output: " << output);
            }
        }
        // pattern match
        auto p = OperatorPattern{pres, node->op_type};
        if (g_fusePattern.find(p) != g_fusePattern.end()) {
            auto fuse_op = g_fusePattern[p];
            vector<string> fuse_inputs{};
            vector<Attribute> fuse_attrs{};
            g.nodes.remove(node);
            string fuse_name = node->name;
            for (auto output : node->inputs_) {
                if (outputOfNode.find(output) != outputOfNode.end()) {
                    auto pre = outputOfNode[output];
                    fuse_name += "_" + pre->name;
                    fuse_inputs.insert(fuse_inputs.end(), pre->inputs_.cbegin(), pre->inputs_.cend());
                    g.nodes.remove(pre);
                } else { // weight
                    fuse_inputs.insert(fuse_inputs.end(), output);
                }
            }
            // 因为合并消除的都是中间变量，不会存储，不需要单独删除
            auto fuse_node = make_shared<Node>(fuse_name, fuse_op, fuse_inputs, node->outputs, fuse_attrs);
            std::ostringstream oss;
            oss << "[opt-fuse operator]: " << p << " => " << fuse_op;
            logger.info(oss.str());
            g.nodes.push_back(fuse_node);

            queue.push_back(fuse_node);
        } else {
            for (auto output : node->inputs_) {
                if (outputOfNode.find(output) != outputOfNode.end()) {
                    auto pre = outputOfNode[output];
                    queue.push_back(pre);
                }
            }
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
            if (outputMap.find(output) != outputMap.end()) {
                queue.push_back(outputMap[output]);
            } else {
                ASSERT_MSG(inputSet.find(output) != inputSet.end() || g.weights.find(output) != g.weights.end(), "unexpected node output: " << output);
            }
        }
    }

    // remove dead node: nodes
    set<string> weight, inputs, outputs;
    for (auto it = g.nodes.begin(); it != g.nodes.end();) {
        if (seen.find(*it) == seen.end()) {
            logger.info("[opt-remove dead node] : " + it->get()->name);
            it = g.nodes.erase(it);
        } else {
            for (auto input : (*it)->inputs_) {
                weight.emplace(input);
            }
            ++it;
        }
    }
    // delete weights & inputs & outputs if only those node use

}

void GraphOptimizer::constFolding(Graph &g, std::set<std::string> skip_ops) {

    unordered_map<string, vector<shared_ptr<Node>>> node_input;
    unordered_map<shared_ptr<Node>, int> node_extra;

    for (auto node : g.nodes) {
        for (auto input : node->inputs_) {
            if (node_input.find(input) != node_input.end()) {
                node_input[input].push_back(node);
            } else {
                node_input[input] = vector{node};
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
            if (node_extra[node] == node->inputs_.size() && skip_ops.find(node->op_type) == skip_ops.end()) {
                // exec
                vector<shared_ptr<Tensor>> input(node->inputs_.size());
                for (int i = 0; i < node->inputs_.size(); ++i) {
                    input[i] = g.weights[node->inputs_[i]];
                }
                vector<shared_ptr<Tensor>> output{};
                OperatorContext ctx(node->attributes);
#ifdef USE_CUDA
                ctx.setStream(nullptr);
#endif
                auto f = M_OP_MAP[node->op_type];
                f(input, output, ctx);
#ifdef USE_CUDA
                CUDA_CHECK(cudaStreamSynchronize(nullptr));
#endif
                assert(output.size() == node->outputs.size());
                auto data_type = input[0]->data_type();
                // add new weight
                for (int i = 0; i < node->outputs.size(); ++i) {
                    q.push(node->outputs[i]);

                    g.weights[node->outputs[i]] = output[i];
                }
                logger.info("[opt-const folding]: calculate node:" + node->name + " and generate new input: " + tostring(node->outputs));
                // remove nodes & weights & inputs
                g.nodes.remove(node);
                for (auto input_name : node->inputs_) {
                    g.weights.erase(input_name);
                }
            }
        }
    }
}

std::unique_ptr<ExecutionGraph> Graph::parse() {
    vector<shared_ptr<ParsedNode>> parsedNodes{};
    for (const auto &n: nodes) {
        parsedNodes.push_back(make_shared<ParsedNode>(n->inputs_, n->outputs, n->op_type, n->attributes));
    }
    vector<pair<int, int>> edges;

    // 一个变量只能被一个节点 “生产”，可以被多个节点 “消费”
    unordered_map<string, shared_ptr<ParsedNode>> allOutputName;
    unordered_map<shared_ptr<ParsedNode>, int> nodeNo;
    unordered_map<int, shared_ptr<ParsedNode>> idxNode;

    int idx = 0;
    for (const auto &n: parsedNodes) {
        nodeNo[n] = idx;
        idxNode[idx] = n;
        idx++;

        for (const auto &item: n->outputs) {
            if (allOutputName.find(item) != allOutputName.end()) {
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
        if (indegree[j] == 0) {
            start.push_back(idxNode[j]);
        }
    }

    return make_unique<ExecutionGraph>(start, weights, inputs, outputs);
}

std::string ParsedNode::tostring() {
    string res = op_type + "(";
    for (const auto &item: inputs) {
        res += (item + ", ");
    }
    return res + ")";
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
    optimizer.constFolding(*this, {OP_EXPAND});
    optimizer.fuseLayers(*this);
    optimizer.constFolding(*this);
}

namespace tensorengine {
    std::unordered_map<OperatorPattern, std::string, OperatorPatternHash> g_fusePattern = {
        {OperatorPattern{set<string>{OP_GEMM}, OP_ADD}, F_OP_MMA},
        {OperatorPattern{set<string>{OP_EXPAND}, OP_GEMM}, F_OP_BATCH_MM},
        {OperatorPattern{set<string>{OP_EXPAND}, F_OP_MMA}, F_OP_BATCH_MMA},
        {OperatorPattern{set<string>{F_OP_BATCH_MM}, OP_ADD}, F_OP_BATCH_MMA},
        {OperatorPattern{set<string>{OP_EXPAND}, OP_ADD}, F_OP_BATCH_ADD},
    };
}
