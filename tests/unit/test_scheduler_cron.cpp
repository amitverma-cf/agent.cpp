#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <chrono>
#include <thread>

namespace {

const char *kWorkspace = "/tmp/agent_test_workspace";

agent::Result<std::string_view> single_shot_transition(agent::Session &session, agent::FlowMemory &conv, void *) {
    auto r = agent::run_turn(session, conv, "go");
    if (!r.ok) return agent::fail<std::string_view>(r.error.code, r.error.message);
    return agent::ok(std::string_view("exit"));
}

} // namespace

TEST_CASE("add_cron fires a one-shot task exactly once", "[scheduler][cron]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .workspace_dir = kWorkspace,
                                       .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = ":memory:"}});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    int fire_count = 0;
    agent::AgentScheduler scheduler;
    auto add_res = scheduler.add_cron(
        "one_shot",
        [](agent::Session &, void *ud) -> agent::Result<void> {
            *static_cast<int *>(ud) += 1;
            return agent::ok();
        },
        &fire_count, std::chrono::milliseconds(0), std::chrono::milliseconds(0));
    REQUIRE(add_res.ok);

    REQUIRE(scheduler.pump(session).ok);
    REQUIRE(fire_count == 1);

    REQUIRE(scheduler.pump(session).ok);
    REQUIRE(fire_count == 1);
}

TEST_CASE("add_cron fires a recurring task more than once", "[scheduler][cron]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .workspace_dir = kWorkspace,
                                       .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = ":memory:"}});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    int fire_count = 0;
    agent::AgentScheduler scheduler;
    auto add_res = scheduler.add_cron(
        "recurring",
        [](agent::Session &, void *ud) -> agent::Result<void> {
            *static_cast<int *>(ud) += 1;
            return agent::ok();
        },
        &fire_count, std::chrono::milliseconds(0), std::chrono::milliseconds(20));
    REQUIRE(add_res.ok);

    REQUIRE(scheduler.pump(session).ok); // fires immediately (delay=0)
    REQUIRE(fire_count == 1);

    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    REQUIRE(scheduler.pump(session).ok);
    REQUIRE(fire_count == 2);
}

TEST_CASE("cancel_cron prevents a scheduled task from firing", "[scheduler][cron]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .workspace_dir = kWorkspace,
                                       .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = ":memory:"}});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    int fire_count = 0;
    agent::AgentScheduler scheduler;
    REQUIRE(scheduler
                .add_cron(
                    "cancel_me",
                    [](agent::Session &, void *ud) -> agent::Result<void> {
                        *static_cast<int *>(ud) += 1;
                        return agent::ok();
                    },
                    &fire_count, std::chrono::milliseconds(0), std::chrono::milliseconds(0))
                .ok);

    auto cancel_res = scheduler.cancel_cron("cancel_me");
    REQUIRE(cancel_res.ok);

    REQUIRE(scheduler.pump(session).ok);
    REQUIRE(fire_count == 0);

    auto cancel_missing = scheduler.cancel_cron("nonexistent");
    REQUIRE_FALSE(cancel_missing.ok);
    REQUIRE(cancel_missing.error.code == agent::ErrorCode::KeyNotFound);
}

TEST_CASE("add_cron rejects a duplicate id", "[scheduler][cron]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .workspace_dir = kWorkspace,
                                       .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = ":memory:"}});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);
    (void)session;

    agent::AgentScheduler scheduler;
    auto ok1 = scheduler.add_cron(
        "dup", [](agent::Session &, void *) -> agent::Result<void> { return agent::ok(); }, nullptr, std::chrono::milliseconds(1000));
    REQUIRE(ok1.ok);

    auto ok2 = scheduler.add_cron(
        "dup", [](agent::Session &, void *) -> agent::Result<void> { return agent::ok(); }, nullptr, std::chrono::milliseconds(1000));
    REQUIRE_FALSE(ok2.ok);
    REQUIRE(ok2.error.code == agent::ErrorCode::InvalidConfig);
}

TEST_CASE("OnCronFire and OnSubAgentComplete events fire during scheduling", "[scheduler][cron][events]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .workspace_dir = kWorkspace,
                                       .memory = {.provider = agent::MemoryProvider::Sqlite, .db_path = ":memory:"}});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    int cron_fires = 0, subagent_completes = 0;
    agent::register_event_hook(
        session, agent::EventType::OnCronFire, [](const agent::Event &, void *ud) { *static_cast<int *>(ud) += 1; }, &cron_fires);
    agent::register_event_hook(
        session, agent::EventType::OnSubAgentComplete, [](const agent::Event &, void *ud) { *static_cast<int *>(ud) += 1; },
        &subagent_completes);

    agent::AgentScheduler scheduler;
    REQUIRE(scheduler
                .add_cron(
                    "noop", [](agent::Session &, void *) -> agent::Result<void> { return agent::ok(); }, nullptr,
                    std::chrono::milliseconds(0), std::chrono::milliseconds(0))
                .ok);

    static const agent::FlowState kStates[] = {
        {.state_id = "go", .on_transition = single_shot_transition},
    };
    agent::Flow flow;
    flow.states = kStates;
    flow.memory.id = "cron_events_flow";
    REQUIRE(scheduler.spawn("worker", std::move(flow)).ok);

    REQUIRE(scheduler.run_until_done(session).ok);

    REQUIRE(cron_fires == 1);
    REQUIRE(subagent_completes == 1);
}
