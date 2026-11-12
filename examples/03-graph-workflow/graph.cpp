// docs/guides/writing-a-graph.md: graphs from C++ builders and from agent.json, scoped state,
// checkpointers, interrupts, time travel (fork) and per-node models from a ModelRegistry.
//
//   guide_graph <engine plugin> [examples/03-graph-workflow/graphs dir] [workspace]
//
// Runs on any text engine; with the mock engine plugin every Agent node replies "echo: <input>".

#include <agent-cpp/agent.hpp>
#include <agent-cpp/store_sqlite.hpp>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

int g_failures = 0;

void check(bool ok, const char *what, const std::string &detail = {}) {
    std::printf("[%s] %s%s%s\n", ok ? "ok" : "FAIL", what, detail.empty() ? "" : ": ", detail.c_str());
    if (!ok) ++g_failures;
}

std::string state_of(const agent::GraphRun &run, std::string_view key) { return std::string(agent::get_state(run.state, key)); }

// A GraphRun on a fresh thread: drops the thread's old checkpoints (Session store), so re-running the
// example starts over instead of continuing from the previous run's state.
agent::GraphRun fresh_run(agent::Session &session, std::string thread_id) {
    agent::delete_graph_thread(session, thread_id);
    agent::GraphRun run;
    run.thread_id = std::move(thread_id);
    return run;
}

// ---- node functions --------------------------------------------------------------------------------

// Function node: a stand-in reviewer that approves on the second round.
agent::Result<void> review(agent::Session &, agent::GraphState &state, void *) {
    const int rounds = std::atoi(std::string(agent::get_state(state, "rounds", "0")).c_str()) + 1;
    agent::set_state(state, "rounds", std::to_string(rounds));
    agent::set_state(state, "verdict", rounds >= 2 ? "APPROVED" : "needs work");
    return agent::ok();
}

// Loop condition, checked after every iteration.
agent::Result<bool> approved(agent::Session &, const agent::GraphState &state, void *) {
    return agent::ok(agent::get_state(state, "verdict") == "APPROVED");
}

// Function node + router: classify, then pick the next node from state.
agent::Result<void> classify(agent::Session &, agent::GraphState &state, void *) {
    const std::string_view request = agent::get_state(state, "request");
    agent::set_state(state, "temp:category", request.find("crash") != std::string_view::npos ? "bug" : "question");
    return agent::ok();
}

agent::Result<std::string_view> route_by_category(agent::Session &, const agent::GraphState &state, void *) {
    const std::string_view category = agent::get_state(state, "temp:category");
    if (category == "bug") return agent::ok(std::string_view("triage_bug"));
    if (category == "question") return agent::ok(std::string_view("answer"));
    return agent::ok(agent::kGraphEnd);
}

// Human in the loop: pause until someone answers, then record the decision.
agent::Result<void> approve(agent::Session &, agent::GraphState &state, void *) {
    if (!agent::has_state(state, "temp:human_input")) {
        agent::request_interrupt(state, "publish this draft?");
        return agent::ok();
    }
    agent::set_state(state, "approval", std::string(agent::get_state(state, "temp:human_input")));
    return agent::ok();
}

agent::Result<void> publish(agent::Session &, agent::GraphState &state, void *) {
    agent::set_state(state, "published", agent::get_state(state, "approval") == "yes" ? "true" : "false");
    agent::set_state(state, "user:last_topic", std::string(agent::get_state(state, "topic"))); // persisted, every thread
    agent::set_state(state, "temp:scratch", "gone after the call");                            // never checkpointed
    return agent::ok();
}

agent::GraphNode agent_node(std::string name, std::string system_prompt, std::string input_key, std::string output_key) {
    return agent::GraphNode{.name = std::move(name),
                            .kind = agent::GraphNodeKind::Agent,
                            .system_prompt = std::move(system_prompt),
                            .input_key = std::move(input_key),
                            .output_key = std::move(output_key)};
}

agent::GraphNode function_node(std::string name, agent::GraphFunctionFn fn) {
    return agent::GraphNode{.name = std::move(name), .kind = agent::GraphNodeKind::Function, .function = fn};
}

// ---- the sections of the guide ---------------------------------------------------------------------

void builders(agent::Session &session) {
    std::printf("-- builders: sequential + loop + parallel\n");
    agent::Graph draft = agent::make_graph(agent_node("draft", "Write a short draft.", "topic", "draft"));
    agent::Graph refine = agent::make_loop("refine", agent::make_graph(function_node("review", review)), 5, approved);
    agent::Graph title = agent::make_graph(agent_node("title", "Write a title.", "draft", "title"));
    agent::Graph summary = agent::make_graph(agent_node("summary", "Summarise.", "draft", "summary"));
    agent::Graph g = agent::make_sequential("article", {draft, refine, agent::make_parallel("publish", {title, summary})});
    agent::GraphRun run = fresh_run(session, "builders");
    agent::set_state(run.state, "topic", "solid-state batteries");
    auto res = agent::run_graph(session, g, run);
    check(res.ok && run.status == agent::GraphRunStatus::Completed, "run_graph completed", res.error.message);
    check(state_of(run, "rounds") == "2", "loop stopped once approved", "rounds=" + state_of(run, "rounds"));
    check(!state_of(run, "title").empty() && !state_of(run, "summary").empty(), "both parallel branches wrote their key");
}

void router(agent::Session &session) {
    std::printf("-- conditional edge\n");
    agent::Graph g;
    g.name = "triage";
    g.entry = "classify";
    g.nodes = {function_node("classify", classify), agent_node("triage_bug", "Ask for a stack trace.", "request", "answer"),
               agent_node("answer", "Answer the question.", "request", "answer")};
    g.edges = {{.from = "classify", .router = route_by_category}};
    agent::GraphRun run = fresh_run(session, "router");
    agent::set_state(run.state, "request", "the app crashes on start");
    auto res = agent::run_graph(session, g, run);
    bool via_bug = false;
    for (const agent::GraphEvent &e : run.events) via_bug |= e.type == agent::GraphEventType::NodeStart && e.node == "triage_bug";
    check(res.ok && via_bug, "routed to triage_bug", res.error.message);
}

agent::Graph approval_graph() {
    return agent::make_sequential("doc", {agent::make_graph(agent_node("write", "Write a short paragraph.", "topic", "draft")),
                                          agent::make_graph(function_node("approve", approve)),
                                          agent::make_graph(function_node("publish_doc", publish))});
}

void checkpoints_and_interrupts(agent::Session &session, const std::string &workspace) {
    std::printf("-- sqlite checkpointer, interrupt + resume, fork\n");
    auto cp = agent::open_checkpointer(agent::store::sqlite_checkpointer(), {.path = workspace + "/runs.sqlite3"});
    check(cp.ok, "open_checkpointer(sqlite)", cp.error.message);
    if (!cp.ok) return;
    const agent::Graph g = approval_graph();
    agent::delete_graph_thread(session, "doc-1", cp.value); // make the example re-runnable
    agent::delete_graph_thread(session, "doc-1-alt", cp.value);

    agent::GraphRun run;
    run.thread_id = "doc-1";
    run.checkpointer = cp.value;
    run.durability = agent::GraphDurability::Sync;
    agent::set_state(run.state, "topic", "local inference");
    auto res = agent::run_graph(session, g, run);
    check(res.ok && run.status == agent::GraphRunStatus::Interrupted && run.interrupt_reason == "publish this draft?",
          "run pauses at approve", run.interrupt_reason);

    // Later (possibly another process): reopen the same file and answer.
    agent::GraphRun resumed;
    resumed.checkpointer = cp.value;
    res = agent::resume_graph(session, g, "doc-1", resumed, "yes");
    check(res.ok && resumed.status == agent::GraphRunStatus::Completed && state_of(resumed, "published") == "true",
          "resume_graph with the answer", res.error.message);
    check(agent::has_state(resumed.state, "user:last_topic") && !agent::has_state(resumed.state, "temp:scratch"),
          "user: kept, temp: dropped");

    auto steps = agent::list_graph_checkpoints(session, "doc-1", cp.value);
    check(steps.ok && steps.value.size() >= 3, "list_graph_checkpoints", std::to_string(steps.value.size()) + " checkpoints");

    // Time travel: continue step 1 (after "write"; "approve" runs next) as a new thread with another topic.
    agent::GraphRun alt;
    alt.checkpointer = cp.value;
    agent::set_state(alt.state, "topic", "cloud inference");
    res = agent::fork_graph_run(session, g, "doc-1", 1, "doc-1-alt", alt);
    check(res.ok && alt.status == agent::GraphRunStatus::Interrupted, "fork_graph_run pauses at approve again", res.error.message);
    agent::GraphRun alt_done;
    alt_done.checkpointer = cp.value;
    res = agent::resume_graph(session, g, "doc-1-alt", alt_done, "no");
    check(res.ok && state_of(alt_done, "published") == "false", "the fork is answered independently", res.error.message);

    auto events = agent::list_graph_events(session, "doc-1-alt", cp.value);
    check(events.ok && !events.value.empty() && events.value.front().type == agent::GraphEventType::Fork,
          "the fork's event log starts with a Fork event");
}

void per_node_models(agent::Session &session, const std::string &engine, const std::string &workspace) {
    std::printf("-- per-node models from a ModelRegistry\n");
    agent::ModelRegistry registry(agent::ModelBudget{.ram_bytes = 8ull << 30});
    for (const char *name : {"small", "large"}) {
        agent::ModelSpec spec;
        spec.name = name;
        spec.config.provider = agent::AiProvider::Engine;
        spec.config.engine = engine;
        spec.config.model = name; // a GGUF path with a real engine
        spec.config.workspace_dir = workspace;
        spec.estimated_bytes = 1ull << 30; // the model file does not exist here, so do not estimate from disk
        auto reg = agent::register_model(registry, spec);
        check(reg.ok, "register_model", reg.error.message);
    }
    agent::GraphNode quick = agent_node("quick", "Answer briefly.", "question", "short");
    quick.model = "small";
    agent::GraphNode deep = agent_node("deep", "Answer in depth.", "question", "long");
    deep.model = "large";
    agent::Graph g = agent::make_sequential("two_models", {agent::make_graph(quick), agent::make_graph(deep)});

    agent::GraphRun run = fresh_run(session, "models");
    run.models = &registry; // resolves GraphNode::model; must outlive the call
    agent::set_state(run.state, "question", "why is the sky blue?");
    auto res = agent::run_graph(session, g, run);
    check(res.ok && !state_of(run, "short").empty() && !state_of(run, "long").empty(), "each node ran on its model", res.error.message);
    check(agent::loaded_models(registry).size() == 2, "both models loaded (idle, evictable)");
}

constexpr const char *kInlineGraph = R"({
  "name": "inline_review",
  "nodes": [
    {"name": "draft", "kind": "agent", "system_prompt": "Write a short draft.", "input_key": "topic", "output_key": "draft"},
    {"name": "refine", "kind": "loop", "body": "review", "max_iterations": 5, "until": {"key": "verdict", "equals": "APPROVED"}},
    {"name": "review", "kind": "function", "function": "review"}
  ],
  "edges": [{"from": "draft", "to": "refine"}]
})";

void from_json(agent::Session &session, const std::string &examples_dir) {
    std::printf("-- agent.json\n");
    agent::register_graph_function("review", review); // before parsing: function nodes are looked up by name
    auto loaded = agent::load_graph_json(session, kInlineGraph);
    check(loaded.ok && loaded.value.warnings.empty(), "load_graph_json (inline)", loaded.error.message);
    if (loaded.ok) {
        agent::GraphRun run = fresh_run(session, "json");
        agent::set_state(run.state, "topic", "edge AI");
        auto res = agent::run_graph(session, loaded.value.graph, run);
        check(res.ok && state_of(run, "rounds") == "2", "the JSON loop stops once approved", res.error.message);
    }
    if (examples_dir.empty()) return;
    for (const char *file : {"router.json", "review_loop.json"}) {
        const std::string path = examples_dir + "/" + file;
        auto example = agent::load_graph_json(session, path);
        check(example.ok, file, example.ok ? "engine hint: " + example.value.engine : example.error.message);
        if (!example.ok) continue;
        agent::GraphRun run = fresh_run(session, file);
        agent::set_state(run.state, "request", "What is 2 + 2?"); // router.json
        agent::set_state(run.state, "task", "Explain what a KV cache is.");
        auto res = agent::run_graph(session, example.value.graph, run);
        check(res.ok && run.status == agent::GraphRunStatus::Completed, "ran", res.error.message);
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <engine plugin> [examples/03-graph-workflow/graphs dir] [workspace]\n", argv[0]);
        return 2;
    }
    const std::string engine = argv[1];
    const std::string examples_dir = argc > 2 ? argv[2] : "";
    const std::string workspace = argc > 3 ? argv[3] : (std::filesystem::temp_directory_path() / "agent-guide-graph").string();
    agent::AgentRuntime runtime;

    agent::Config cfg;
    cfg.provider = agent::AiProvider::Engine;
    cfg.engine = engine;
    cfg.workspace_dir = workspace;
    cfg.memory.store = agent::store::sqlite_store(); // user:/app: state persists in <workspace>/memory/db.sqlite3
    auto sess = agent::init(cfg);
    check(sess.ok, "init", sess.error.message);
    if (!sess.ok) return 1;
    agent::Session session = std::move(sess.value);

    builders(session);
    router(session);
    checkpoints_and_interrupts(session, workspace);
    per_node_models(session, engine, workspace);
    from_json(session, examples_dir);

    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
