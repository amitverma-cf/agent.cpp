#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>

namespace {

const char *kWorkspace = "/tmp/agent_test_workspace";

agent::Result<std::string_view> single_shot_transition(agent::Session &session, agent::FlowMemory &conv, void *) {
    auto r = agent::run_turn(session, conv, "go");
    if (!r.ok) return agent::fail<std::string_view>(r.error.code, r.error.message);
    return agent::ok(std::string_view("exit"));
}

} // namespace

TEST_CASE("OnSubAgentComplete fires during scheduling", "[scheduler][events]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock,
                                       .workspace_dir = kWorkspace});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    int subagent_completes = 0;
    agent::register_event_hook(
        session, agent::EventType::OnSubAgentComplete, [](const agent::Event &, void *ud) { *static_cast<int *>(ud) += 1; },
        &subagent_completes);

    agent::AgentScheduler scheduler;

    static const agent::FlowState kStates[] = {
        {.state_id = "go", .on_transition = single_shot_transition},
    };
    agent::Flow flow;
    flow.states = kStates;
    flow.memory.id = "events_flow";
    REQUIRE(scheduler.spawn("worker", std::move(flow)).ok);

    REQUIRE(scheduler.run_until_done(session).ok);

    REQUIRE(subagent_completes == 1);
}
