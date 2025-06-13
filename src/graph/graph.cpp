#include "../../include/graph.h"
#include "queue"

using namespace tensorengine;
using namespace std;

// notice never save ParsedGraph*
void GraphOptimizer::fuseLayers(ParsedGraph* pg) {

}

// notice never save ParsedGraph*
void GraphOptimizer::removeDeadNodes(ParsedGraph* pg) {

}

std::unique_ptr<ParsedGraph> Graph::parse() {
    vector<shared_ptr<ParsedNode>> parsedNodes(nodes.size());
    for (const auto &n: nodes) {
        parsedNodes.push_back(make_shared<ParsedNode>(n->inputs, n->outputs, n->op_type, n->attributes));
    }
    vector<pair<int, int>> edges;

    // 一个变量只能被一个节点 “生产”（produce），可以被多个节点 “消费”（consume）
    unordered_map<string, shared_ptr<ParsedNode>> allOutputName;
    unordered_map<shared_ptr<ParsedNode>, int> nodeNo;
    unordered_map<int, shared_ptr<ParsedNode>> idxNode;

    int i = 0;
    for (const auto &n: parsedNodes) {
        nodeNo[n] = i;
        idxNode[i] = n;
        i++;

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
        if (indegree[i] == 0) {
            start.push_back(idxNode[i]);
        }
    }

    return make_unique<ParsedGraph>(start, weights, inputs, outputs);
}

void ParsedGraph::opt() {
    GraphOptimizer optimizer;
    optimizer.fuseLayers(this);
    optimizer.removeDeadNodes(this);
}

