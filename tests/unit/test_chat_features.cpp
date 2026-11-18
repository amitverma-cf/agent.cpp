// Notes and chat search, installed models by capability, skills and MCP servers added by name, and workflows saved by name.
#include "catalog_browse.hpp"
#include "chat_memory.hpp"
#include "chat_models.hpp"
#include "cli_core.hpp"
#include "home.hpp"
#include "mcp_config.hpp"
#include "workflows.hpp"

#include <agent-cpp/agent.hpp>
#include <agent-cpp/hub.hpp>
#include <agent-cpp/store_sqlite.hpp>
#include <catch_amalgamated.hpp>
#include <filesystem>
#include <fstream>

using namespace agent::cli;
namespace fs = std::filesystem;

namespace {

std::string fresh(const char *name) {
    const fs::path dir = fs::temp_directory_path() / name;
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir.string();
}

void write_all(const fs::path &p, const std::string &text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << text;
}

Context make_context(const std::string &root) {
    Context ctx;
    ctx.config_dir = root + "/config";
    ctx.home = home_layout(root + "/home");
    ctx.has_home = true;
    create_home_layout(ctx.home);
    return ctx;
}

agent::Session memory_session(const std::string &dir) {
    auto s = agent::init({.provider = agent::AiProvider::Mock,
                          .workspace_dir = dir,
                          .memory = {.store = agent::store::sqlite_store(), .db_path = dir + "/mem.sqlite3"},
                          .auto_default_tools = false});
    REQUIRE(s.ok);
    return std::move(s.value);
}

} // namespace

TEST_CASE("notes: remember, forget by name and by words, honest answers", "[chat_features][notes]") {
    auto session = memory_session(fresh("agent_notes_a"));
    REQUIRE(note_keep(session, "favourite colour", "the user likes green").ok);
    REQUIRE(note_keep(session, "dog", "the dog is called Biscuit").ok);
    CHECK(note_keys(session).size() == 2);

    bool removed = true;
    std::string answer = note_forget(session, "cat", "", removed); // no such note
    CHECK_FALSE(removed);
    CHECK(answer.find("nothing was removed") != std::string::npos);
    CHECK(note_keys(session).size() == 2);

    answer = note_forget(session, "", "Biscuit", removed); // by words
    CHECK(removed);
    CHECK(answer.find("removed the note 'dog'") != std::string::npos);
    CHECK(note_keys(session).size() == 1);
    CHECK(recall_text(session, "dog", "", "").find("Biscuit") == std::string::npos);

    answer = note_forget(session, "favourite colour", "", removed); // by name
    CHECK(removed);
    CHECK(note_keys(session).empty());
}

TEST_CASE("notes: forget refuses to guess between equal matches", "[chat_features][notes]") {
    auto session = memory_session(fresh("agent_notes_b"));
    REQUIRE(note_keep(session, "trip one", "going to Paris").ok);
    REQUIRE(note_keep(session, "trip two", "going to Paris").ok);
    bool removed = true;
    const std::string answer = note_forget(session, "", "Paris", removed);
    CHECK_FALSE(removed);
    CHECK(answer.find("several notes match") != std::string::npos);
    CHECK(note_keys(session).size() == 2);
}

TEST_CASE("recall searches the saved chats and says how many it searched", "[chat_features][notes]") {
    auto session = memory_session(fresh("agent_notes_c"));
    std::string index;
    const char *topics[] = {"planning a birthday party for my sister", "fixing the python script that parses invoices"};
    for (int i = 0; i < 2; ++i) {
        agent::Conversation c;
        c.id = "chat-" + std::to_string(i);
        c.history.push_back({.role = "user", .content = topics[i]});
        c.history.push_back({.role = "assistant", .content = "Sure, let's work on it."});
        REQUIRE(agent::save_conversation(session, c).ok);
        index = add_conversation_id(index, c.id);
    }
    REQUIRE(agent::store_memory(session, "cli:conversations", index).ok);

    size_t searched = 0;
    const auto hits = search_chats(session, "birthday", "chat-9", searched);
    CHECK(searched == 2);
    REQUIRE(hits.size() == 1);
    CHECK(hits[0].id == "chat-0");

    const std::string text = recall_text(session, "", "invoices", "chat-9");
    CHECK(text.find("Searched 2 earlier chats") != std::string::npos);
    CHECK(text.find("chat-1") != std::string::npos);

    const std::string earlier = recall_text(session, "", "", "chat-1"); // nothing to go on: the latest chats, not the open one
    CHECK(earlier.find("Searched 1 earlier chat of") != std::string::npos);
}

TEST_CASE("models are told apart by what they can do", "[chat_features][models]") {
    const std::string dir = fresh("agent_models_a");
    write_all(dir + "/qwen.gguf", "not a real gguf");
    std::vector<agent::hub::CatalogEntry> entries;
    entries.push_back({.name = "talker", .repo = "o/qwen-chat", .file = "qwen-chat-q4.gguf", .path = dir + "/qwen.gguf"});
    entries.push_back(
        {.name = "reader", .repo = "o/qwen-vl", .file = "qwen-vl-q4.gguf", .path = dir + "/qwen.gguf", .mmproj = dir + "/mm.gguf"});
    entries.push_back(
        {.name = "voice", .repo = "audio/PocketTTS-GGUF", .file = "PocketTTS-GGUF/english/pocket-tts.gguf", .path = dir + "/p.gguf"});
    entries.push_back({.name = "cloud", .repo = "api", .file = "x/y", .path = dir + "/cloud.api.json"});
    REQUIRE(agent::hub::save_catalog(dir + "/models.json", entries).ok);

    const auto models = installed_models(dir + "/models.json");
    REQUIRE(models.size() == 4);
    CHECK(models[1].capability == "vision");
    CHECK(models[2].capability == "tts");
    CHECK(models[3].remote);
    CHECK(can_chat(models[0]));
    CHECK(can_chat(models[1]));
    CHECK_FALSE(can_chat(models[2]));
    CHECK(can_chat(models[3])); // a remote model shows in Settings > Model

    const std::string prompt = installed_models_prompt(models);
    CHECK(prompt.find("tts (text to speech): voice") != std::string::npos);
    CHECK(prompt.find("cloud (remote)") != std::string::npos);
    CHECK(prompt.find("Never invent a model name") != std::string::npos);

    bool found = false;
    CHECK(model_for_task(models, "tts", "invented-voice", found) == "voice"); // a guessed name becomes the installed model
    CHECK(found);
    CHECK(model_for_task(models, "tts", "voice", found) == "voice");
    CHECK(model_for_task(models, "asr", "whisper-xl", found).empty());
    CHECK_FALSE(found); // nothing installed for speech recognition
}

TEST_CASE("beginner_error drops command hints", "[chat_features][models]") {
    const std::string text = beginner_error("engine 'api' is not installed. Run `agent setup` or `agent engine add <plugin>`.\ncannot open "
                                            "the file\nTry: agent model pull x\n");
    CHECK(text.find("agent setup") == std::string::npos);
    CHECK(text.find("agent model pull") == std::string::npos);
    CHECK(text.find("cannot open the file") != std::string::npos);
}

TEST_CASE("MCP servers are added by name, package or address, switched and removed", "[chat_features][mcp]") {
    const std::string root = fresh("agent_mcp_a");
    const Context ctx = make_context(root);
    const std::string ws = root + "/ws";
    fs::create_directories(ws);

    auto spec = parse_mcp_spec("@modelcontextprotocol/server-filesystem");
    REQUIRE(spec.ok);
    CHECK(spec.value.command == "npx");
    CHECK(spec.value.name == "filesystem");
    auto uv = parse_mcp_spec("uvx mcp-server-time");
    REQUIRE(uv.ok);
    CHECK(uv.value.command == "uvx");
    CHECK(uv.value.name == "time");
    auto web = parse_mcp_spec("https://mcp.example.com/mcp");
    REQUIRE(web.ok);
    CHECK(web.value.url == "https://mcp.example.com/mcp");
    CHECK(web.value.name == "example");

    REQUIRE(mcp_add_curated(ctx, ws, "filesystem").ok);
    REQUIRE(mcp_add_spec(ctx, ws, uv.value).ok);
    CHECK_FALSE(mcp_add_curated(ctx, ws, "no-such-server").ok);

    std::vector<std::string> warnings;
    const auto servers = collect_mcp_servers(ctx, warnings);
    REQUIRE(servers.size() == 2);
    const auto fs_server = std::find_if(servers.begin(), servers.end(), [](const auto &s) { return s.name == "filesystem"; });
    REQUIRE(fs_server != servers.end());
    CHECK(fs_server->args.back().find("ws") != std::string::npos); // the folder it may use is the project

    REQUIRE(mcp_set_enabled(ctx, ws, "time", false).ok);
    CHECK(collect_mcp_servers(ctx, warnings).size() == 1);
    const auto listed = list_mcp(ctx, ws);
    CHECK(std::any_of(listed.begin(), listed.end(), [](const McpInstalled &m) { return m.name == "time" && !m.enabled; }));
    REQUIRE(mcp_set_enabled(ctx, ws, "time", true).ok);
    CHECK(collect_mcp_servers(ctx, warnings).size() == 2);
    REQUIRE(mcp_remove_named(ctx, ws, "time").ok);
    CHECK(collect_mcp_servers(ctx, warnings).size() == 1);
    CHECK_FALSE(mcp_remove_named(ctx, ws, "time").ok);
}

TEST_CASE("an MCP server is added from the JSON block of its page", "[chat_features][mcp]") {
    const std::string root = fresh("agent_mcp_json");
    const Context ctx = make_context(root);
    const std::string ws = root + "/ws";
    fs::create_directories(ws);

    // The README shape: a whole mcpServers object, with a key kept out of the file as a secret reference.
    REQUIRE(mcp_add_json_text(ctx, ws,
                              R"({"mcpServers":{"github":{"command":"npx","args":["-y","@modelcontextprotocol/server-github"],"env":{"GITHUB_PERSONAL_ACCESS_TOKEN":"${secret:github}"}}}})")
                .ok);
    // One entry without a name: the name comes from the command.
    REQUIRE(mcp_add_json_text(ctx, ws, R"({"command":"uvx","args":["mcp-server-time"]})").ok);
    // One entry with a name, reached over HTTP with a header.
    REQUIRE(mcp_add_json_text(ctx, ws, R"({"name":"docs","url":"https://example.com/mcp","headers":{"Authorization":"Bearer ${secret:docs}"}})").ok);
    CHECK_FALSE(mcp_add_json_text(ctx, ws, "[1]").ok);
    CHECK_FALSE(mcp_add_json_text(ctx, ws, R"({"args":["x"]})").ok); // no command and no address

    std::vector<std::string> warnings;
    const auto servers = collect_mcp_servers(ctx, warnings);
    REQUIRE(servers.size() == 3);
    const auto github = std::find_if(servers.begin(), servers.end(), [](const auto &s) { return s.name == "github"; });
    REQUIRE(github != servers.end());
    CHECK(github->args.back() == "@modelcontextprotocol/server-github");
    CHECK(std::any_of(servers.begin(), servers.end(), [](const auto &s) { return s.name == "time"; }));
    const auto docs = std::find_if(servers.begin(), servers.end(), [](const auto &s) { return s.name == "docs"; });
    REQUIRE(docs != servers.end());
    CHECK(docs->url == "https://example.com/mcp");
}

TEST_CASE("a skill is named by id, skills.sh address, npx line or GitHub folder", "[chat_features][skills]") {
    CHECK(skill_id_from_text("anthropics/skills/pdf") == "anthropics/skills/pdf");
    CHECK(skill_id_from_text("  https://skills.sh/anthropics/skills/pdf/ ") == "anthropics/skills/pdf");
    CHECK(skill_id_from_text("npx skills add anthropics/skills --skill pdf") == "anthropics/skills/pdf");
    CHECK(skill_id_from_text("https://github.com/anthropics/skills/tree/main/skills/pdf") == "anthropics/skills/pdf");
}

TEST_CASE("a project's .agents/mcp.json is found from the workspace", "[chat_features][mcp]") {
    const std::string root = fresh("agent_mcp_b");
    Context ctx = make_context(root);
    const std::string ws = root + "/ws";
    write_all(ws + "/.agents/mcp.json", R"({"mcpServers":{"echo":{"command":"python3","args":["server.py"]}}})");
    std::vector<std::string> warnings;
    CHECK(collect_mcp_servers(ctx, warnings).empty()); // the context was made before the folder was a project
    ctx.project_root = find_project_root(ws);
    const auto servers = collect_mcp_servers(ctx, warnings);
    REQUIRE(servers.size() == 1);
    CHECK(servers[0].name == "echo");
    CHECK(list_mcp(ctx, ws).size() == 1); // list_mcp finds the project on its own
}

TEST_CASE("skills are listed, switched off and on, and removed", "[chat_features][skills]") {
    const std::string root = fresh("agent_skills_a");
    const Context ctx = make_context(root);
    const std::string ws = root + "/ws";
    write_all(ctx.home.skills + "/commit-helper/SKILL.md",
              "---\nname: commit-helper\ndescription: Writes short commit messages\n---\nUse imperative mood.\n");
    write_all(ws + "/.agents/skills/local-one/SKILL.md", "---\nname: local-one\ndescription: Only here\n---\nbody\n");

    auto skills = list_skills(ctx, ws);
    REQUIRE(skills.size() == 2);
    CHECK(skills[0].name == "commit-helper");
    CHECK(skills[0].where == "every project");
    CHECK(skills[0].description == "Writes short commit messages");
    CHECK(skills[1].where == "this project");

    REQUIRE(skill_set_enabled(ctx, ws, "commit-helper", false).ok);
    CHECK_FALSE(fs::exists(ctx.home.skills + "/commit-helper/SKILL.md"));
    CHECK_FALSE(list_skills(ctx, ws)[0].enabled);
    REQUIRE(skill_set_enabled(ctx, ws, "commit-helper", true).ok);
    CHECK(list_skills(ctx, ws)[0].enabled);

    REQUIRE(skill_remove_named(ctx, ws, "local-one").ok);
    CHECK(list_skills(ctx, ws).size() == 1);
    CHECK_FALSE(skill_remove_named(ctx, ws, "../escape").ok);

    CHECK(skill_dirs(ctx, ws).size() == 2);
}

TEST_CASE("workflows are saved by name, listed, resolved and removed", "[chat_features][workflows]") {
    const std::string root = fresh("agent_workflows_a");
    const Context ctx = make_context(root);
    // Three steps: a prompt, a loop with a route out of it, parallel branches.
    const std::string graph = R"({
      "description": "daily report",
      "entry": "gather",
      "nodes": [
        {"name": "gather", "kind": "agent", "prompt": "List three facts", "output_key": "facts"},
        {"name": "polish", "kind": "loop", "body": "draft", "max_iterations": 2},
        {"name": "draft", "kind": "agent", "prompt": "Improve the text", "input_key": "facts", "output_key": "facts"},
        {"name": "publish", "kind": "parallel", "branches": ["mail", "archive"]},
        {"name": "mail", "kind": "agent", "prompt": "Write the mail", "input_key": "facts"},
        {"name": "archive", "kind": "agent", "prompt": "Write the archive note", "input_key": "facts"}
      ],
      "edges": [
        {"from": "gather", "to": "polish"},
        {"from": "polish", "to": "publish"},
        {"from": "publish", "to": "__end__"}
      ]})";
    auto saved = workflow_save(ctx, "daily-report", graph);
    REQUIRE(saved.ok);
    CHECK(saved.value.find("daily-report") != std::string::npos);
    CHECK(fs::exists(root + "/home/graphs/daily-report.json"));

    CHECK_FALSE(workflow_save(ctx, "bad name", graph).ok);
    CHECK_FALSE(workflow_save(ctx, "broken", R"({"nodes":[{"name":"a","kind":"agent"}],"edges":[{"from":"a","to":"nowhere"}]})").ok);
    CHECK_FALSE(fs::exists(root + "/home/graphs/broken.json"));

    const auto list = workflow_list(ctx);
    REQUIRE(list.size() == 1);
    CHECK(list[0].name == "daily-report");
    CHECK(list[0].description == "daily report");
    CHECK(list[0].nodes == 6);
    // Compared as paths: on Windows the file is named with backslashes.
    CHECK(fs::path(resolve_graph_arg(ctx, "daily-report")) == fs::path(root + "/home/graphs/daily-report.json"));
    CHECK(fs::path(resolve_graph_arg(ctx, "daily-report.json")) == fs::path(root + "/home/graphs/daily-report.json"));
    CHECK(resolve_graph_arg(ctx, "other") == "other");
    CHECK(workflows_prompt(ctx).find("daily-report (daily report)") != std::string::npos);

    REQUIRE(workflow_remove(ctx, "daily-report").ok);
    CHECK(workflow_list(ctx).empty());
    CHECK(workflows_prompt(ctx).empty());
}
