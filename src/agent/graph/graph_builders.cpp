#include <agent-cpp/agent.hpp>
#include <algorithm>
#include <utility>

namespace agent {

void set_state(GraphState &state, std::string_view key, std::string value) {
    auto it = state.values.find(key);
    if (it != state.values.end()) it->second = std::move(value);
    else state.values.emplace(std::string(key), std::move(value));
    state.written.emplace_back(key);
}

std::string_view get_state(const GraphState &state, std::string_view key, std::string_view fallback) {
    auto it = state.values.find(key);
    return it == state.values.end() ? fallback : std::string_view(it->second);
}

bool has_state(const GraphState &state, std::string_view key) { return state.values.find(key) != state.values.end(); }

void request_interrupt(GraphState &state, std::string_view reason) {
    state.interrupt_requested = true;
    state.interrupt_reason = std::string(reason);
}

namespace {

// Points every node that would end the sub-graph at target, unless an inner builder already did.
void chain_exits(Graph &g, std::string_view target) {
    for (auto &node : g.nodes)
        if (node.exit_to.empty()) node.exit_to = std::string(target);
}

void append(Graph &dst, Graph &&src) {
    for (auto &node : src.nodes) dst.nodes.push_back(std::move(node));
    for (auto &edge : src.edges) dst.edges.push_back(std::move(edge));
    dst.max_steps = std::max(dst.max_steps, src.max_steps);
}

} // namespace

Graph make_graph(GraphNode node) {
    Graph g;
    g.name = node.name;
    g.entry = node.name;
    g.nodes.push_back(std::move(node));
    return g;
}

Graph make_sequential(std::string name, std::vector<Graph> steps) {
    Graph g;
    g.name = std::move(name);
    if (steps.empty()) return g;
    g.entry = steps.front().entry;
    for (size_t i = 0; i < steps.size(); ++i) {
        if (i + 1 < steps.size()) chain_exits(steps[i], steps[i + 1].entry);
        append(g, std::move(steps[i]));
    }
    return g;
}

Graph make_parallel(std::string name, std::vector<Graph> branches) {
    Graph g;
    g.name = name;
    g.entry = name;
    GraphNode fan;
    fan.name = name;
    fan.kind = GraphNodeKind::Parallel;
    for (const auto &b : branches) fan.branches.push_back(b.entry);
    g.nodes.push_back(std::move(fan));
    for (auto &b : branches) {
        chain_exits(b, name); // reaching the Parallel node ends the branch
        append(g, std::move(b));
    }
    return g;
}

Graph make_loop(std::string name, Graph body, int max_iterations, GraphUntilFn until, void *user_data) {
    Graph g;
    g.name = name;
    g.entry = name;
    GraphNode ctl;
    ctl.name = name;
    ctl.kind = GraphNodeKind::Loop;
    ctl.loop_body = body.entry;
    ctl.max_iterations = max_iterations;
    ctl.until = until;
    ctl.user_data = user_data;
    g.nodes.push_back(std::move(ctl));
    chain_exits(body, name); // the end of an iteration returns to the controller
    append(g, std::move(body));
    return g;
}

} // namespace agent
