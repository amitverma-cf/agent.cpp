#include <agent-cpp/agent.hpp>
#include <catch_amalgamated.hpp>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::string make_skills_dir(const char *name) {
    std::string dir = "/tmp/agent_test_skills_" + std::string(name);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    return dir;
}

void write_skill(const std::string &skills_dir, const std::string &skill_name, const std::string &frontmatter_name,
                 const std::string &description, const std::string &body_extra = "") {
    std::string skill_dir = skills_dir + "/" + skill_name;
    std::filesystem::create_directories(skill_dir);
    std::ofstream f(skill_dir + "/SKILL.md");
    f << "---\n";
    f << "name: " << frontmatter_name << "\n";
    f << "description: " << description << "\n";
    f << "---\n";
    f << "# " << frontmatter_name << "\n\n" << body_extra << "\n";
}

const agent::Tool *find_tool(const agent::Session &session, std::string_view name) {
    for (const auto &t : session.config.tools)
        if (t.name == name) return &t;
    return nullptr;
}

} // namespace

TEST_CASE("load_skills_dir populates session.skills from SKILL.md frontmatter", "[skills]") {
    std::string skills_dir = make_skills_dir("basic");
    write_skill(skills_dir, "pdf-tools", "pdf-tools", "Use when working with PDF files.");
    write_skill(skills_dir, "xlsx-tools", "xlsx-tools", "Use when working with spreadsheets.");

    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    auto res = agent::load_skills_dir(session, skills_dir);
    REQUIRE(res.ok);
    REQUIRE(session.skills.size() == 2);
    REQUIRE(session.skills.at("pdf-tools").description == "Use when working with PDF files.");
    REQUIRE(session.skills.at("xlsx-tools").description == "Use when working with spreadsheets.");
}

TEST_CASE("read_skill_body returns the full SKILL.md content on demand", "[skills]") {
    std::string skills_dir = make_skills_dir("body");
    write_skill(skills_dir, "demo", "demo", "A demo skill.", "Detailed instructions go here.");

    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    REQUIRE(agent::load_skills_dir(session, skills_dir).ok);

    auto body_res = agent::read_skill_body(session, "demo");
    REQUIRE(body_res.ok);
    REQUIRE(body_res.value.find("Detailed instructions go here.") != std::string::npos);
    REQUIRE(body_res.value.find("---") == 0);
}

TEST_CASE("read_skill_body returns KeyNotFound for an unknown skill", "[skills]") {
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    auto res = agent::read_skill_body(session, "nonexistent");
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::KeyNotFound);
}

TEST_CASE("load_skills_dir registers a use_skill tool that loads the full body", "[skills][tools]") {
    std::string skills_dir = make_skills_dir("tool");
    write_skill(skills_dir, "demo", "demo", "A demo skill.", "Full body content here.");

    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws"});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    REQUIRE(agent::load_skills_dir(session, skills_dir).ok);
    REQUIRE(find_tool(session, "use_skill") != nullptr);

    // Dispatch via execute_tool (not the Tool::callback directly): execute_tool refreshes
    // session.session_ptr_cell to point at this Session before invoking, which the tool's
    // callback relies on. Calling the raw callback directly would use a stale cell left
    // over from the temporary Session inside agent::init(), before it was moved from.
    auto res = agent::execute_tool(session, "use_skill", R"({"name":"demo"})");
    REQUIRE(res.ok);
    REQUIRE(res.value.find("Full body content here.") != std::string::npos);

    auto missing_res = agent::execute_tool(session, "use_skill", R"({"name":"nope"})");
    REQUIRE_FALSE(missing_res.ok);
    REQUIRE(missing_res.error.code == agent::ErrorCode::KeyNotFound);
}

TEST_CASE("Config::skills_dir auto-loads skills at init()", "[skills]") {
    std::string skills_dir = make_skills_dir("autoload");
    write_skill(skills_dir, "auto", "auto", "Auto-loaded skill.");

    auto session_result =
        agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = "/tmp/agent_test_ws", .skills_dir = skills_dir});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    REQUIRE(session.skills.count("auto") == 1);
    REQUIRE(find_tool(session, "use_skill") != nullptr);
    REQUIRE(session.tools_system_prompt.find("auto") != std::string::npos);
}
