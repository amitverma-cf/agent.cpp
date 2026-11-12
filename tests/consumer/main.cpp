// Smoke test for an installed agent-cpp: one Mock turn with the SQLite store from
// agent-cpp::store_sqlite. Exits non-zero on any failure.
#include <agent-cpp/agent.hpp>
#include <agent-cpp/store_sqlite.hpp>
#include <cstdio>
#include <filesystem>
#include <utility>

int main() {
    agent::AgentRuntime runtime;

    const auto workspace = std::filesystem::temp_directory_path() / "agent-cpp-consumer";
    agent::Config cfg;
    cfg.provider = agent::AiProvider::Mock;
    cfg.workspace_dir = workspace.string();
    cfg.memory.store = agent::store::sqlite_store();
    cfg.memory.db_path = ":memory:";

    auto sess = agent::init(cfg);
    if (!sess.ok) {
        std::fprintf(stderr, "init failed: %s\n", sess.error.message.c_str());
        return 1;
    }
    agent::Session session = std::move(sess.value);

    agent::Conversation conv;
    conv.id = "consumer";
    auto r = agent::run_turn(session, conv, "hello");
    if (!r.ok) {
        std::fprintf(stderr, "run_turn failed: %s\n", r.error.message.c_str());
        return 1;
    }
    std::printf("reply: %s\n", r.value.c_str());

    if (auto put = agent::store_memory(session, "greeting", r.value); !put.ok) {
        std::fprintf(stderr, "store_memory failed: %s\n", put.error.message.c_str());
        return 1;
    }
    auto got = agent::retrieve_memory(session, "greeting");
    if (!got.ok || got.value != r.value) {
        std::fprintf(stderr, "retrieve_memory mismatch\n");
        return 1;
    }
    std::printf("sqlite store round trip: %s\n", got.value.c_str());
    return 0;
}
