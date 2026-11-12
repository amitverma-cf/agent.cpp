#include <algorithm>
#include <agent-cpp/agent.hpp>
#include <atomic>
#include <catch_amalgamated.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr uint64_t kMiB = 1024ull * 1024ull;
const char *kWorkspace = "/tmp/agent_test_models_ws";

// A sparse fake model file of exactly `bytes` bytes (the mock engine never reads it).
std::string fake_model(const std::string &name, uint64_t bytes) {
    const fs::path dir = fs::path(kWorkspace) / "fake_models";
    fs::create_directories(dir);
    const fs::path p = dir / name;
    { std::ofstream(p, std::ios::binary | std::ios::trunc); }
    fs::resize_file(p, bytes);
    return p.string();
}

std::atomic<int> g_loads{0};

agent::Result<void> count_load(agent::Session &, const agent::ModelSpec &, void *) {
    g_loads.fetch_add(1);
    return agent::ok();
}

agent::ModelSpec mock_spec(const std::string &name, uint64_t bytes, bool pinned = false) {
    agent::ModelSpec spec;
    spec.name = name;
    spec.config.provider = agent::AiProvider::Engine;
    spec.config.engine = AGENT_TEST_MOCK_ENGINE_PATH;
    spec.config.model = fake_model(name + ".bin", bytes);
    spec.config.workspace_dir = kWorkspace;
    spec.config.auto_default_tools = false;
    spec.pinned = pinned;
    spec.on_load = count_load;
    return spec;
}

std::vector<std::string> loaded_names(const agent::ModelRegistry &registry) {
    std::vector<std::string> out;
    for (const auto &m : agent::loaded_models(registry)) out.push_back(m.name);
    return out;
}

bool is_loaded(const agent::ModelRegistry &registry, std::string_view name) {
    for (const auto &m : agent::loaded_models(registry))
        if (m.name == name) return true;
    return false;
}

// Acquire, then drop the lease at once (marks the model most recently used).
void touch(agent::ModelRegistry &registry, std::string_view name) {
    auto lease = agent::acquire_model(registry, name);
    INFO(lease.error.message);
    REQUIRE(lease.ok);
}

} // namespace

TEST_CASE("estimate_model_memory: mock = file size, overrides, extra files, gpu_layers", "[models]") {
    auto spec = mock_spec("est", 3 * kMiB);
    auto est = agent::estimate_model_memory(spec);
    INFO(est.error.message);
    REQUIRE(est.ok);
    REQUIRE(est.value.file_bytes == 3 * kMiB);
    REQUIRE(est.value.ram_bytes == 3 * kMiB);
    REQUIRE(est.value.vram_bytes == 0);

    // Files named in engine_options (mmproj, ...) are added; non-file strings are ignored.
    std::string proj = fake_model("est-mmproj.bin", 1 * kMiB);
    std::replace(proj.begin(), proj.end(), '\\', '/'); // a Windows path must not put backslashes into the JSON below
    spec.config.engine_options = R"({"mmproj":")" + proj + R"(","mode":"generate","nested":{"x":[")" + proj + R"("]}})";
    est = agent::estimate_model_memory(spec);
    REQUIRE(est.ok);
    REQUIRE(est.value.ram_bytes == 4 * kMiB);

    // gpu_layers > 0 moves the estimate to VRAM.
    spec.config.engine_options = R"({"gpu_layers": 99})";
    est = agent::estimate_model_memory(spec);
    REQUIRE(est.ok);
    REQUIRE(est.value.vram_bytes == 3 * kMiB);
    REQUIRE(est.value.ram_bytes == 0);

    // estimated_bytes skips the files entirely.
    spec.config.engine_options.clear();
    spec.config.model = "/nonexistent/model.bin";
    spec.estimated_bytes = 42;
    est = agent::estimate_model_memory(spec);
    REQUIRE(est.ok);
    REQUIRE(est.value.ram_bytes == 42);

    spec.estimated_bytes = 0;
    est = agent::estimate_model_memory(spec);
    REQUIRE_FALSE(est.ok);
    REQUIRE(est.error.code == agent::ErrorCode::FilesystemError);

    spec.config.engine_options = "{not json";
    spec.estimated_bytes = 1;
    REQUIRE(agent::estimate_model_memory(spec).error.code == agent::ErrorCode::InvalidConfig);
}

TEST_CASE("estimate_model_memory: llama adds KV cache and overhead, split GGUFs are summed", "[models]") {
    agent::ModelSpec spec;
    spec.name = "llama-est";
    spec.config.provider = agent::AiProvider::Engine;
    spec.config.engine = "llama";
    spec.config.context_window = 1024;
    spec.config.max_parallel_subagents = 2;
    spec.config.model = fake_model("split-00001-of-00003.gguf", 32 * kMiB);
    fake_model("split-00002-of-00003.gguf", 32 * kMiB);
    fake_model("split-00003-of-00003.gguf", 16 * kMiB);
    auto est = agent::estimate_model_memory(spec);
    REQUIRE(est.ok);
    const uint64_t file = 80 * kMiB;
    REQUIRE(est.value.file_bytes == file);
    REQUIRE(est.value.ram_bytes == file + 1024 * 2 * (file / 32768) + 64 * kMiB);
}

TEST_CASE("process_rss_bytes reports the resident set", "[models]") {
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
    REQUIRE(agent::process_rss_bytes() > 0);
#endif
}

TEST_CASE("ModelRegistry loads lazily and keeps idle models loaded", "[models]") {
    agent::ModelRegistry registry;
    REQUIRE(agent::register_model(registry, mock_spec("lazy", 1 * kMiB)).ok);
    REQUIRE(agent::has_model(registry, "lazy"));
    REQUIRE_FALSE(agent::has_model(registry, "other"));
    REQUIRE(agent::loaded_models(registry).empty());

    g_loads = 0;
    {
        auto lease = agent::acquire_model(registry, "lazy");
        REQUIRE(lease.ok);
        REQUIRE(lease.value);
        REQUIRE(lease.value.name == "lazy");
        auto models = agent::loaded_models(registry);
        REQUIRE(models.size() == 1);
        REQUIRE(models[0].leases == 1);
        REQUIRE(models[0].ram_bytes == 1 * kMiB);
        REQUIRE(models[0].loaded_at_ms > 0);

        agent::Conversation memory;
        memory.id = "lazy";
        auto reply = agent::run_turn(*lease.value, memory, "hi");
        REQUIRE(reply.ok);
        REQUIRE(reply.value == "echo: hi");

        // A second lease shares the Session.
        auto again = agent::acquire_model(registry, "lazy");
        REQUIRE(again.ok);
        REQUIRE(again.value.session == lease.value.session);
        REQUIRE(agent::loaded_models(registry)[0].leases == 2);
        agent::release_model(again.value);
        agent::release_model(again.value); // idempotent
        REQUIRE(agent::loaded_models(registry)[0].leases == 1);
    }
    auto models = agent::loaded_models(registry);
    REQUIRE(models.size() == 1);
    REQUIRE(models[0].leases == 0);
    touch(registry, "lazy");
    REQUIRE(g_loads == 1);

    auto missing = agent::acquire_model(registry, "nope");
    REQUIRE_FALSE(missing.ok);
    REQUIRE(missing.error.code == agent::ErrorCode::KeyNotFound);
    REQUIRE(agent::register_model(registry, agent::ModelSpec{}).error.code == agent::ErrorCode::InvalidConfig);
}

TEST_CASE("ModelRegistry evicts the least recently used idle model", "[models]") {
    agent::ModelRegistry registry(agent::ModelBudget{.ram_bytes = 25 * kMiB});
    for (const char *n : {"a", "b", "c"}) REQUIRE(agent::register_model(registry, mock_spec(n, 10 * kMiB)).ok);

    touch(registry, "a");
    touch(registry, "b");
    touch(registry, "a"); // b is now the least recently used
    REQUIRE(loaded_names(registry) == std::vector<std::string>{"a", "b"});

    touch(registry, "c");
    REQUIRE(loaded_names(registry) == std::vector<std::string>{"c", "a"});

    touch(registry, "b"); // evicts a
    REQUIRE(loaded_names(registry) == std::vector<std::string>{"b", "c"});

    // Lowering the budget evicts down to it, LRU first.
    agent::set_budget(registry, agent::ModelBudget{.ram_bytes = 15 * kMiB});
    REQUIRE(loaded_names(registry) == std::vector<std::string>{"b"});
    REQUIRE(agent::get_budget(registry).ram_bytes == 15 * kMiB);
}

TEST_CASE("ModelRegistry never evicts pinned models", "[models]") {
    agent::ModelRegistry registry(agent::ModelBudget{.ram_bytes = 25 * kMiB});
    REQUIRE(agent::register_model(registry, mock_spec("pinned", 10 * kMiB, true)).ok);
    REQUIRE(agent::register_model(registry, mock_spec("b", 10 * kMiB)).ok);
    REQUIRE(agent::register_model(registry, mock_spec("c", 10 * kMiB)).ok);

    touch(registry, "pinned"); // least recently used, but pinned
    touch(registry, "b");
    touch(registry, "c");
    REQUIRE(is_loaded(registry, "pinned"));
    REQUIRE(is_loaded(registry, "c"));
    REQUIRE_FALSE(is_loaded(registry, "b"));
    REQUIRE(agent::loaded_models(registry).size() == 2);

    agent::set_budget(registry, agent::ModelBudget{.ram_bytes = 5 * kMiB}); // pinned alone is over budget
    REQUIRE(loaded_names(registry) == std::vector<std::string>{"pinned"});
    auto r = agent::acquire_model(registry, "b");
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.error.code == agent::ErrorCode::OutOfMemory);
    REQUIRE(r.error.message.find("pinned") != std::string::npos);

    // unload_model still unloads a pinned model.
    REQUIRE(agent::unload_model(registry, "pinned").ok);
    REQUIRE(agent::loaded_models(registry).empty());
}

TEST_CASE("ModelRegistry never evicts leased models", "[models]") {
    agent::ModelRegistry registry(agent::ModelBudget{.ram_bytes = 15 * kMiB});
    REQUIRE(agent::register_model(registry, mock_spec("held", 10 * kMiB)).ok);
    REQUIRE(agent::register_model(registry, mock_spec("other", 10 * kMiB)).ok);

    auto held = agent::acquire_model(registry, "held");
    REQUIRE(held.ok);
    auto other = agent::acquire_model(registry, "other");
    REQUIRE_FALSE(other.ok);
    REQUIRE(other.error.code == agent::ErrorCode::OutOfMemory);
    REQUIRE(other.error.message.find("in use") != std::string::npos);
    REQUIRE(loaded_names(registry) == std::vector<std::string>{"held"});

    agent::release_model(held.value);
    other = agent::acquire_model(registry, "other"); // now held is idle and gets evicted
    REQUIRE(other.ok);
    REQUIRE(loaded_names(registry) == std::vector<std::string>{"other"});
}

TEST_CASE("ModelRegistry: a model larger than the budget fails unless overcommit is allowed", "[models]") {
    agent::ModelRegistry registry(agent::ModelBudget{.ram_bytes = 5 * kMiB});
    REQUIRE(agent::register_model(registry, mock_spec("big", 10 * kMiB)).ok);
    REQUIRE(agent::register_model(registry, mock_spec("small", 2 * kMiB)).ok);
    g_loads = 0;
    auto r = agent::acquire_model(registry, "big");
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.error.code == agent::ErrorCode::OutOfMemory);
    REQUIRE(r.error.message.find("big") != std::string::npos);
    REQUIRE(agent::loaded_models(registry).empty());
    REQUIRE(g_loads == 0);

    // Overcommit: idle models are still evicted first, then it loads over budget.
    touch(registry, "small");
    agent::set_budget(registry, agent::ModelBudget{.ram_bytes = 5 * kMiB, .allow_overcommit = true});
    r = agent::acquire_model(registry, "big");
    REQUIRE(r.ok);
    REQUIRE(loaded_names(registry) == std::vector<std::string>{"big"});

    // With the big one leased, a second model loads beside it.
    auto small = agent::acquire_model(registry, "small");
    REQUIRE(small.ok);
    REQUIRE(agent::loaded_models(registry).size() == 2);
}

TEST_CASE("ModelRegistry: concurrent acquires of one model load it once", "[models]") {
    agent::ModelRegistry registry;
    REQUIRE(agent::register_model(registry, mock_spec("shared", 1 * kMiB)).ok);
    g_loads = 0;
    std::atomic<int> ok_count{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&registry, &ok_count, i] {
            auto lease = agent::acquire_model(registry, "shared");
            if (!lease.ok) return;
            std::lock_guard<std::mutex> turn(*lease.value.turn_mutex);
            agent::Conversation memory;
            memory.id = "shared-" + std::to_string(i);
            auto reply = agent::run_turn(*lease.value, memory, "t" + std::to_string(i));
            if (reply.ok && reply.value == "echo: t" + std::to_string(i)) ok_count.fetch_add(1);
        });
    }
    for (auto &t : threads) t.join();
    REQUIRE(ok_count == 8);
    REQUIRE(g_loads == 1);
    REQUIRE(agent::loaded_models(registry)[0].leases == 0);
}

TEST_CASE("ModelRegistry: concurrent acquires of different models wait for room", "[models]") {
    // Room for two of the four models; each thread holds its lease briefly. wait_ms lets acquirers
    // wait for a release instead of failing.
    agent::ModelRegistry registry(agent::ModelBudget{.ram_bytes = 20 * kMiB, .wait_ms = 20000});
    for (int i = 0; i < 4; ++i) REQUIRE(agent::register_model(registry, mock_spec("m" + std::to_string(i), 10 * kMiB)).ok);
    std::atomic<int> ok_count{0};
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 12; ++i) {
        threads.emplace_back([&, i] {
            auto lease = agent::acquire_model(registry, "m" + std::to_string(i % 4));
            if (!lease.ok) {
                errors.fetch_add(1);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            ok_count.fetch_add(1);
        });
    }
    for (auto &t : threads) t.join();
    REQUIRE(errors == 0);
    REQUIRE(ok_count == 12);
    uint64_t total = 0;
    for (const auto &m : agent::loaded_models(registry)) total += m.ram_bytes;
    REQUIRE(total <= 20 * kMiB);
}

TEST_CASE("ModelRegistry: unload, reload and re-register", "[models]") {
    agent::ModelRegistry registry;
    REQUIRE(agent::register_model(registry, mock_spec("u", 1 * kMiB)).ok);
    g_loads = 0;
    REQUIRE(agent::unload_model(registry, "u").ok); // not loaded: ok
    REQUIRE(agent::unload_model(registry, "nope").error.code == agent::ErrorCode::KeyNotFound);

    auto lease = agent::acquire_model(registry, "u");
    REQUIRE(lease.ok);
    auto busy = agent::unload_model(registry, "u");
    REQUIRE_FALSE(busy.ok);
    REQUIRE(busy.error.code == agent::ErrorCode::InvalidConfig);
    REQUIRE(agent::register_model(registry, mock_spec("u", 2 * kMiB)).error.code == agent::ErrorCode::InvalidConfig);

    agent::release_model(lease.value);
    REQUIRE(agent::unload_model(registry, "u").ok);
    REQUIRE(agent::loaded_models(registry).empty());
    touch(registry, "u");
    REQUIRE(g_loads == 2);

    // Re-registering an idle loaded model unloads it; the next acquire uses the new spec.
    REQUIRE(agent::register_model(registry, mock_spec("u", 2 * kMiB)).ok);
    REQUIRE(agent::loaded_models(registry).empty());
    touch(registry, "u");
    REQUIRE(agent::loaded_models(registry)[0].ram_bytes == 2 * kMiB);
}

TEST_CASE("ModelRegistry: load failures release the reservation", "[models]") {
    agent::ModelRegistry registry(agent::ModelBudget{.ram_bytes = 10 * kMiB});
    auto bad = mock_spec("bad", 1 * kMiB);
    bad.config.model = "fail"; // the mock engine refuses it
    bad.estimated_bytes = 8 * kMiB;
    REQUIRE(agent::register_model(registry, bad).ok);
    auto r = agent::acquire_model(registry, "bad");
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.error.message.find("bad") != std::string::npos);
    REQUIRE(agent::loaded_models(registry).empty());

    auto refusing = mock_spec("refusing", 1 * kMiB);
    refusing.on_load = [](agent::Session &, const agent::ModelSpec &, void *) {
        return agent::fail(agent::ErrorCode::InvalidConfig, "no");
    };
    REQUIRE(agent::register_model(registry, refusing).ok);
    r = agent::acquire_model(registry, "refusing");
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.error.code == agent::ErrorCode::InvalidConfig);
    REQUIRE(agent::loaded_models(registry).empty());

    // The whole budget is still available.
    REQUIRE(agent::register_model(registry, mock_spec("fits", 10 * kMiB)).ok);
    touch(registry, "fits");
}

TEST_CASE("Graph Agent nodes run on GraphNode::model through GraphRun::models", "[models][graph]") {
    agent::Config home_cfg;
    home_cfg.provider = agent::AiProvider::Mock;
    home_cfg.workspace_dir = kWorkspace;
    auto home = agent::init(home_cfg);
    REQUIRE(home.ok);

    agent::ModelRegistry registry(agent::ModelBudget{.ram_bytes = 15 * kMiB});
    REQUIRE(agent::register_model(registry, mock_spec("writer", 10 * kMiB)).ok);
    REQUIRE(agent::register_model(registry, mock_spec("critic", 10 * kMiB)).ok);

    auto loaded = agent::parse_graph_json(R"({
        "name": "two_models", "entry": "write",
        "nodes": [
            {"name": "write", "kind": "agent", "model": "writer", "prompt": "draft", "output_key": "draft"},
            {"name": "review", "kind": "agent", "model": "critic", "input_key": "draft", "output_key": "review"}
        ],
        "edges": [{"from": "write", "to": "review"}]
    })");
    INFO(loaded.error.message);
    REQUIRE(loaded.ok);
    REQUIRE(loaded.value.warnings.empty());
    REQUIRE(loaded.value.graph.nodes[0].model == "writer");

    agent::GraphRun run;
    run.thread_id = "models-graph";
    run.durability = agent::GraphDurability::Exit;
    run.models = &registry;
    auto res = agent::run_graph(home.value, loaded.value.graph, run);
    INFO(res.error.message);
    REQUIRE(res.ok);
    REQUIRE(run.status == agent::GraphRunStatus::Completed);
    REQUIRE(agent::get_state(run.state, "draft") == "echo: draft");
    REQUIRE(agent::get_state(run.state, "review") == "echo: echo: draft");
    // The budget fits one model: the writer was evicted for the critic.
    REQUIRE(loaded_names(registry) == std::vector<std::string>{"critic"});

    agent::GraphRun no_registry;
    no_registry.thread_id = "models-graph-none";
    auto r2 = agent::run_graph(home.value, loaded.value.graph, no_registry);
    REQUIRE_FALSE(r2.ok);
    REQUIRE(r2.error.code == agent::ErrorCode::InvalidConfig);
}

#ifdef AGENT_TEST_LLAMA_ENGINE_PATH
// AGENT_TEST_LLAMA_TOOL_MODEL=<gguf, e.g. qwen2.5-0.5b-instruct-q4_k_m.gguf> plus a second, much
// smaller GGUF in AGENT_TEST_LLAMA_TINY_MODEL (default /tmp/models/stories260K.gguf).
TEST_CASE("ModelRegistry switches between two llama models under a one-model budget", "[models][llama][model]") {
    const char *big_path = std::getenv("AGENT_TEST_LLAMA_TOOL_MODEL");
    if (!big_path) SKIP("AGENT_TEST_LLAMA_TOOL_MODEL not set");
    const char *tiny_env = std::getenv("AGENT_TEST_LLAMA_TINY_MODEL");
    const std::string tiny_path = tiny_env ? tiny_env : "/tmp/models/stories260K.gguf";
    if (!fs::exists(tiny_path)) SKIP("no tiny GGUF at " + tiny_path + " (set AGENT_TEST_LLAMA_TINY_MODEL)");

    auto llama = [](std::string name, std::string path) {
        agent::ModelSpec spec;
        spec.name = std::move(name);
        spec.config.provider = agent::AiProvider::Engine;
        spec.config.engine = AGENT_TEST_LLAMA_ENGINE_PATH;
        spec.config.engine_isolation = agent::EngineIsolation::InProcess; // RSS must show the model
        spec.config.model = std::move(path);
        spec.config.workspace_dir = kWorkspace;
        spec.config.context_window = 512;
        spec.config.max_tokens = 8;
        spec.config.temperature = 0.0f;
        spec.config.max_parallel_subagents = 1;
        spec.config.auto_default_tools = false;
        return spec;
    };
    auto big = llama("big", big_path);
    auto tiny = llama("tiny", tiny_path);
    auto big_est = agent::estimate_model_memory(big);
    auto tiny_est = agent::estimate_model_memory(tiny);
    REQUIRE(big_est.ok);
    REQUIRE(tiny_est.ok);
    REQUIRE(big_est.value.ram_bytes > tiny_est.value.ram_bytes);

    // Room for the big model or the tiny one, not both.
    agent::ModelRegistry registry(agent::ModelBudget{.ram_bytes = big_est.value.ram_bytes + tiny_est.value.ram_bytes / 2});
    REQUIRE(agent::register_model(registry, big).ok);
    REQUIRE(agent::register_model(registry, tiny).ok);

    const uint64_t rss_before = agent::process_rss_bytes();
    uint64_t rss_big = 0;
    {
        auto lease = agent::acquire_model(registry, "big");
        INFO(lease.error.message);
        REQUIRE(lease.ok);
        agent::Conversation memory;
        memory.id = "registry-big";
        auto reply = agent::run_turn(*lease.value, memory, "Say hi.");
        INFO(reply.error.message);
        REQUIRE(reply.ok);
        rss_big = agent::process_rss_bytes();
    }
    REQUIRE(loaded_names(registry) == std::vector<std::string>{"big"});

    {
        auto lease = agent::acquire_model(registry, "tiny");
        INFO(lease.error.message);
        REQUIRE(lease.ok);
        agent::Conversation memory;
        memory.id = "registry-tiny";
        REQUIRE(agent::run_turn(*lease.value, memory, "Once upon a time").ok);
    }
    REQUIRE(loaded_names(registry) == std::vector<std::string>{"tiny"});
    const uint64_t rss_tiny = agent::process_rss_bytes();

    const uint64_t file = fs::file_size(big_path);
    INFO("rss before " << rss_before / kMiB << " MiB, with big " << rss_big / kMiB << " MiB, after switch " << rss_tiny / kMiB
                       << " MiB, big file " << file / kMiB << " MiB");
    // Evicting the big model gives back most of its weights.
    REQUIRE(rss_big > rss_tiny);
    REQUIRE(rss_big - rss_tiny > file / 2);
}
#endif
