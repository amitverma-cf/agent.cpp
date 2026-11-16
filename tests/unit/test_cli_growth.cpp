// Timers, self-written skills, the plugin gateway, the memory index and the crash guard of the CLI.

#include "chat_memory.hpp"
#include "crash_guard.hpp"
#include "memory_index.hpp"
#include "plugins.hpp"
#include "skill_learn.hpp"
#include "timers.hpp"

#include <catch_amalgamated.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>

using namespace agent;
using namespace agent::cli;
namespace fs = std::filesystem;

namespace {

fs::path fresh_dir(const char *name) {
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path() / "agent_test_growth" / name;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return dir;
}

void write_file(const fs::path &p, const std::string &text) {
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    out << text;
}

std::string read_file(const fs::path &p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool has(const std::string &text, const std::string &part) { return text.find(part) != std::string::npos; }

// A stand-in embedding model: 8 numbers from the letters of the text, so texts with the same words land close together.
std::vector<std::vector<float>> fake_embed(const std::vector<std::string> &texts) {
    std::vector<std::vector<float>> out;
    for (const std::string &t : texts) {
        std::vector<float> v(8, 0.01f);
        for (const char c : t) v[static_cast<unsigned char>(c) % 8] += 1.0f;
        out.push_back(std::move(v));
    }
    return out;
}

} // namespace

TEST_CASE("timers: add, list, remove and what is due", "[cli][timers]") {
    const std::string home = fresh_dir("timers").string();
    const int64_t now = 1'800'000'000;
    bool failed = true;
    std::string a = timer_tool(home, "/p1", R"({"action":"add","prompt":"stretch","in_minutes":20})", now, failed);
    CHECK_FALSE(failed);
    CHECK(has(a, "t1"));
    CHECK(has(a, "in 20 minutes"));
    timer_tool(home, "/p1", R"({"action":"add","prompt":"check mail","every_minutes":60})", now, failed);
    CHECK_FALSE(failed);
    timer_tool(home, "/p2", R"({"action":"add","prompt":"other project","in_minutes":1})", now, failed);
    CHECK_FALSE(failed);

    CHECK(has(timer_tool(home, "/p1", R"({"action":"list"})", now, failed), "stretch"));
    CHECK_FALSE(has(timer_tool(home, "/p1", R"({"action":"list"})", now, failed), "other project"));

    // Nothing is due yet; after 25 minutes the one-shot fires and is gone, the repeating one is not due until the hour.
    std::vector<Timer> timers = load_timers(home);
    REQUIRE(timers.size() == 3);
    CHECK(take_due_timers(timers, "/p1", now + 10).empty());
    auto due = take_due_timers(timers, "/p1", now + 25 * 60);
    REQUIRE(due.size() == 1);
    CHECK(due[0].prompt == "stretch");
    CHECK(timers.size() == 2);
    // A repeating timer missed three times fires once and moves past now.
    due = take_due_timers(timers, "/p1", now + 3 * 3600 + 10);
    REQUIRE(due.size() == 1);
    CHECK(due[0].prompt == "check mail");
    for (const Timer &t : timers)
        if (t.prompt == "check mail") CHECK(t.next_epoch > now + 3 * 3600 + 10);
    // "*" takes the timers of every project.
    due = take_due_timers(timers, "*", now + 4 * 3600);
    CHECK(due.size() == 2);

    timer_tool(home, "/p1", R"({"action":"remove","id":"t2"})", now, failed);
    CHECK_FALSE(failed);
    timer_tool(home, "/p1", R"({"action":"remove","id":"nope"})", now, failed);
    CHECK(failed);
    timer_tool(home, "/p1", R"({"action":"add","prompt":"no time given"})", now, failed);
    CHECK(failed);
    timer_tool(home, "/p1", R"({"action":"add","prompt":"bad clock","at":"25:99"})", now, failed);
    CHECK(failed);
    timer_tool(home, "/p1", R"({"action":"add","prompt":"morning","at":"09:00","repeat":true})", now, failed);
    CHECK_FALSE(failed);

    CHECK(seconds_until_clock("9:5", now) > 0);
    CHECK(seconds_until_clock("9:5", now) <= 24 * 3600);
    CHECK(seconds_until_clock("nine", now) == -1);
    // The file survives a round trip.
    const auto again = timers_from_json(timers_to_json(load_timers(home)));
    CHECK(again.size() == load_timers(home).size());
}

TEST_CASE("skill_learn: a new skill, a repair with a lesson, bad input", "[cli][skills]") {
    const fs::path skills = fresh_dir("skills");
    bool failed = true;
    std::string a = skill_learn(skills.string(), R"({"name":"deploy-site","description":"Deploy the site","steps":"1. build\n2. upload"})",
                                "2026-11-16", failed);
    CHECK_FALSE(failed);
    const fs::path file = skills / "deploy-site" / "SKILL.md";
    REQUIRE(fs::exists(file));
    std::string doc = read_file(file);
    CHECK(has(doc, "name: deploy-site"));
    CHECK(has(doc, "description: Deploy the site"));
    CHECK(has(doc, "2. upload"));
    CHECK_FALSE(has(doc, "## Lessons"));

    // A lesson alone keeps the steps.
    skill_learn(skills.string(), R"({"name":"deploy-site","lesson":"the build needs NODE_ENV=production"})", "2026-11-17", failed);
    CHECK_FALSE(failed);
    doc = read_file(file);
    CHECK(has(doc, "2. upload"));
    CHECK(has(doc, "- 2026-11-17: the build needs NODE_ENV=production"));

    // New steps replace the old ones and the earlier lesson stays.
    skill_learn(skills.string(),
                R"({"name":"deploy-site","steps":"1. set NODE_ENV\n2. build\n3. upload","lesson":"upload needs the token"})", "2026-11-18",
                failed);
    CHECK_FALSE(failed);
    doc = read_file(file);
    CHECK(has(doc, "1. set NODE_ENV"));
    CHECK(has(doc, "NODE_ENV=production"));
    CHECK(has(doc, "- 2026-11-18: upload needs the token"));
    CHECK(skill_description(doc) == "Deploy the site");
    CHECK(has(skill_steps(doc), "3. upload"));
    CHECK_FALSE(has(skill_steps(doc), "Lessons"));

    skill_learn(skills.string(), R"({"name":"Bad Name","description":"x","steps":"y"})", "2026-11-18", failed);
    CHECK(failed);
    skill_learn(skills.string(), R"({"name":"new-one","description":"only a description"})", "2026-11-18", failed);
    CHECK(failed);
    skill_learn(skills.string(), R"({"name":"deploy-site"})", "2026-11-18", failed);
    CHECK(failed);
    skill_learn(skills.string(), R"({"name":"../escape","description":"x","steps":"y"})", "2026-11-18", failed);
    CHECK(failed);
}

TEST_CASE("plugins: manifests are read, bad ones are reported", "[cli][plugins]") {
    const fs::path root = fresh_dir("plugins");
    write_file(root / "greeter" / "plugin.json",
               R"({"name":"greeter","description":"Greetings","runtime":"python","entry":"main.py","timeout_seconds":20,
                   "tools":[{"name":"hello","description":"Say hello","parameters":{"type":"object","properties":{"who":{"type":"string"}}}},
                            {"name":"fail","description":"Always fails"}]})");
    write_file(root / "greeter" / "main.py",
               "import json, sys\ncall = json.load(sys.stdin)\nif call['tool'] == 'fail':\n    print('nope')\n    sys.exit(3)\n"
               "print('hello ' + call['arguments'].get('who', 'you') + ' from ' + call['workspace'])\n");
    write_file(root / "broken" / "plugin.json", "{not json");
    write_file(root / "noentry" / "plugin.json", R"({"name":"noentry","runtime":"bun","entry":"missing.js","tools":[]})");
    write_file(root / "badruntime" / "plugin.json", R"({"name":"badruntime","runtime":"ruby","entry":"a.rb","tools":[]})");

    std::vector<std::string> problems;
    const auto tools = load_plugins({root.string(), (root / "does-not-exist").string()}, problems);
    REQUIRE(tools.size() == 2);
    CHECK(problems.size() == 3);
    const PluginTool *hello = nullptr, *fail_tool = nullptr;
    for (const PluginTool &t : tools) (t.tool == "hello" ? hello : fail_tool) = &t;
    REQUIRE(hello);
    REQUIRE(fail_tool);
    CHECK(hello->full_name == "greeter_hello");
    CHECK(has(hello->description, "plugin greeter"));
    CHECK(has(hello->schema, "who"));
    CHECK(hello->timeout_seconds == 20);
    CHECK(has(fail_tool->schema, "object"));

    HomeLayout home; // no private runtimes: "python" falls back to python3 on the PATH
    std::string why;
    const auto command = plugin_command(*hello, home, why);
    REQUIRE(command.size() == 2);
    CHECK(command[0] == "python3");

    PluginTool bun_tool = *hello;
    bun_tool.runtime = "bun";
    CHECK(plugin_command(bun_tool, home, why).empty());
    CHECK(has(why, "not installed"));

    if (std::system("python3 --version > /dev/null 2>&1") == 0) {
        bool failed = true;
        const std::string out = run_plugin_tool(*hello, home, R"({"who":"Ada"})", "/work/space", failed);
        CHECK_FALSE(failed);
        CHECK(has(out, "hello Ada from /work/space"));
        const std::string bad = run_plugin_tool(*fail_tool, home, "{}", "/w", failed);
        CHECK(failed);
        CHECK(has(bad, "nope"));
    }
}

TEST_CASE("memory index: words alone, words and vectors, erase, a second run", "[cli][memory_index]") {
    const fs::path ws = fresh_dir("index");
    {
        auto opened = MemoryIndex::open(ws.string(), 0); // no embedding model
        REQUIRE(opened.ok);
        MemoryIndex &index = opened.value;
        REQUIRE(index
                    .put({{"note:diet", "diet: the user is vegetarian"},
                          {"note:dog", "dog: the dog is called Biscuit"},
                          {"chat:c1:0", "user: how do I bake sourdough bread"}},
                         nullptr)
                    .ok);
        auto hits = index.search("what is my dog called", 3, nullptr);
        REQUIRE_FALSE(hits.empty());
        CHECK(hits[0].id == "note:dog");
        CHECK(index.search("quantum chromodynamics", 3, nullptr).empty()); // shares no word: nothing, not noise
        CHECK(index.has("note:diet"));
        REQUIRE(index.erase("note:dog").ok);
        CHECK_FALSE(index.has("note:dog"));
        for (const MemoryHit &h : index.search("dog Biscuit", 3, nullptr)) CHECK(h.id != "note:dog");
        CHECK(memory_hit_chat("chat:c1:0") == "c1");
        CHECK(memory_hit_chat("note:diet").empty());
    }
    {
        // A second run: what was indexed is remembered, so the same texts are not embedded again.
        auto opened = MemoryIndex::open(ws.string(), 0);
        REQUIRE(opened.ok);
        CHECK(opened.value.has("note:diet"));
        CHECK_FALSE(opened.value.has("note:dog"));
    }
    {
        // With an embedding model the table has real vectors (its own table: nothing from the words-only one is mixed in).
        auto opened = MemoryIndex::open(ws.string(), 8);
        REQUIRE(opened.ok);
        MemoryIndex &index = opened.value;
        int embedded = 0;
        const EmbedFn counting = [&](const std::vector<std::string> &texts) {
            embedded += static_cast<int>(texts.size());
            return fake_embed(texts);
        };
        REQUIRE(index.put({{"note:a", "aaaa aaaa aaaa"}, {"note:b", "bbbb bbbb bbbb"}}, counting).ok);
        CHECK(embedded == 2);
        REQUIRE(index.put({{"note:a", "aaaa aaaa aaaa"}}, counting).ok); // unchanged: not embedded again
        CHECK(embedded == 2);
        auto hits = index.search("aaaa", 2, counting);
        REQUIRE_FALSE(hits.empty());
        CHECK(hits[0].id == "note:a");
        // The model is gone: no vectors for the query, so the caller is told there is nothing (it falls back to words itself).
        CHECK(index.search("aaaa", 2, nullptr).empty());
        // A model that returns the wrong number of vectors is an error, not a silent bad index.
        const EmbedFn broken = [](const std::vector<std::string> &) { return std::vector<std::vector<float>>{}; };
        CHECK_FALSE(index.put({{"note:c", "cccc"}}, broken).ok);
    }
}

TEST_CASE("crash guard: a recorded crash is found again and can be forgotten", "[cli][crash_guard]") {
    const fs::path cache = fresh_dir("crash");
    crash_guard_install(cache.string());
    const std::string model = "/models/some/model.gguf";
    CHECK_FALSE(crashed_before(model));
    const std::string marker = crash_marker_path(model);
    REQUIRE_FALSE(marker.empty());
    CHECK(marker != crash_marker_path("/models/other.gguf"));
    { CrashScope scope(model); } // a scope that ends without a crash leaves nothing behind
    CHECK_FALSE(crashed_before(model));
    write_file(marker, model); // what the signal handler does
    CHECK(crashed_before(model));
    crash_forget(model);
    CHECK_FALSE(crashed_before(model));
    CHECK(crash_marker_path("").empty());
}

TEST_CASE("memory: the notes block in front of a stored message is not part of the passage", "[cli][memory_index]") {
    CHECK(without_notes("[Notes kept from earlier chats; use them when they matter]\n- pet: a dog\n[End of notes]\n\nhello there") ==
          "hello there");
    CHECK(without_notes("plain message") == "plain message");
    CHECK(without_notes("[Notes kept from earlier chats; never closed") == "[Notes kept from earlier chats; never closed");
    CHECK(without_notes("[Notes kept from earlier chats]\n[End of notes]\n\n").empty());
}
