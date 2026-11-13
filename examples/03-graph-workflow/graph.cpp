// Two agents in a row: a planner and a writer that continues the planner's conversation.
// Needs models/qwen2.5-0.5b-instruct-q4_k_m.gguf and the llama engine. The same graphs as JSON are in graphs/ (run them with `agent run`).

#include <agent-cpp/agent.hpp>
#include <cstdio>
#include <string>

int main() {
    agent::AgentRuntime runtime;

    agent::Config config;
    config.auto_default_tools = false; // these agents only write text; leave out the file and terminal tools
    auto session = agent::open_session(agent::engines::llama, "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", config);
    if (!session.ok) return std::fprintf(stderr, "open: %s\n", session.error.message.c_str()), 1;

    agent::GraphNode plan{.name = "plan",
                          .kind = agent::GraphNodeKind::Agent,
                          .system_prompt = "You are a careful planner.",
                          .prompt = "List three steps to learn to bake bread.",
                          .history_key = "bread"};
    agent::GraphNode write{.name = "write",
                           .kind = agent::GraphNodeKind::Agent,
                           .prompt = "Turn the plan into one short paragraph.",
                           .output_key = "result",
                           .history_key = "bread"}; // the same history_key continues the planner's conversation

    agent::Graph graph = agent::make_sequential("bread", {agent::make_graph(plan), agent::make_graph(write)});

    agent::GraphRun run;
    run.thread_id = "bread-1";
    auto done = agent::run_graph(session.value, graph, run);
    if (!done.ok) return std::fprintf(stderr, "graph: %s\n", done.error.message.c_str()), 1;

    std::printf("%s\n", std::string(agent::get_state(run.state, "result")).c_str());
    return 0;
}
