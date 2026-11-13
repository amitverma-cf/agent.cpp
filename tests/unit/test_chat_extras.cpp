#include "chat_extras.hpp"

#include <catch_amalgamated.hpp>
#include <filesystem>
#include <fstream>

using namespace agent::cli;
namespace fs = std::filesystem;

namespace {
std::string fresh_dir(const char *name) {
    const fs::path dir = fs::temp_directory_path() / name;
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir.string();
}
void write_all(const fs::path &p, const std::string &text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << text;
}
} // namespace

TEST_CASE("chat extras: @file mentions", "[chat_extras]") {
    const std::string ws = fresh_dir("agent_mentions_ws");
    write_all(fs::path(ws) / "src" / "a.txt", "alpha\nbeta\n");
    write_all(fs::path(ws) / "bin.dat", std::string("x\0y", 3));
    std::vector<std::string> attached;
    const std::string out = expand_mentions("Explain @src/a.txt please, and @missing.txt and @bin.dat. mail me@example.com", ws, &attached);
    CHECK(out.find("Explain @src/a.txt please") == 0);
    CHECK(out.find("File src/a.txt:\n```\nalpha\nbeta\n```") != std::string::npos);
    REQUIRE(attached.size() == 1);
    CHECK(attached[0] == "src/a.txt");
    CHECK(out.find("missing.txt:") == std::string::npos); // not a file: left as typed
    CHECK(out.find("example.com") != std::string::npos);  // an address is not a mention
    CHECK(expand_mentions("no mentions here", ws) == "no mentions here");
    // A sentence ending in the path.
    CHECK(expand_mentions("see @src/a.txt.", ws).find("File src/a.txt") != std::string::npos);
    // Each file is added once.
    const std::string twice = expand_mentions("@src/a.txt and @src/a.txt", ws);
    CHECK(twice.find("File src/a.txt") == twice.rfind("File src/a.txt"));
}

TEST_CASE("chat extras: the text of a web page", "[chat_extras]") {
    const std::string text = html_to_text(
        "<html><head><style>p{color:red}</style><script>var x = '<b>';</script></head><body><h1>Title &amp; more</h1><p>First   para.</p>"
        "<!-- hidden --><div>Second <b>bold</b> line</div><ul><li>one</li><li>two</li></ul></body></html>");
    CHECK(text.find("Title & more") == 0);
    CHECK(text.find("First para.") != std::string::npos);
    CHECK(text.find("Second bold line") != std::string::npos);
    CHECK(text.find("one\ntwo") != std::string::npos);
    CHECK(text.find("color") == std::string::npos);
    CHECK(text.find("var x") == std::string::npos);
    CHECK(text.find("hidden") == std::string::npos);
    CHECK(html_to_text("").empty());
}

TEST_CASE("chat extras: tool arguments for a person", "[chat_extras]") {
    const std::string args = pretty_arguments(
        R"({"path":"a/b.txt","content":"line one\nline two\nline three\nline four\nline five\nline six","create_dirs":true})");
    CHECK(args.find("path: a/b.txt") == 0);
    CHECK(args.find("content: line one\n    line two") != std::string::npos);
    CHECK(args.find("...") != std::string::npos); // more than four lines: cut
    CHECK(args.find("create_dirs: true") != std::string::npos);
    CHECK(pretty_arguments("not json").find("not json") == 0);
    CHECK(pretty_arguments(R"({"command":"ls -la"})") == "command: ls -la");
}

TEST_CASE("chat extras: thinking blocks are hidden unless shown", "[chat_extras]") {
    ThinkFilter f;
    std::string out;
    for (const char *piece : {"Hello ", "<thi", "nk>I should ", "reason</th", "ink>\nAnswer."}) out += filter_think(f, piece);
    out += flush_think(f);
    CHECK(out == "Hello Answer.");
    ThinkFilter shown;
    shown.show = true;
    CHECK(filter_think(shown, "<think>x</think>y") == "<think>x</think>y");
    ThinkFilter odd;
    std::string kept = filter_think(odd, "a < b <");
    kept += flush_think(odd);
    CHECK(kept == "a < b <"); // a possible tag start that never became one is text
}

TEST_CASE("chat extras: modes, permissions file, conversation index", "[chat_extras]") {
    ChatMode mode = ChatMode::Ask;
    CHECK(parse_chat_mode("PLAN", mode));
    CHECK(mode == ChatMode::Plan);
    CHECK(std::string(chat_mode_name(mode)) == "plan");
    CHECK_FALSE(parse_chat_mode("yolo", mode));
    CHECK(tool_changes_things("write_file"));
    CHECK(tool_changes_things("run_command"));
    CHECK_FALSE(tool_changes_things("read_file"));
    CHECK_FALSE(tool_changes_things("web_fetch"));

    const std::string ws = fresh_dir("agent_perm_ws");
    CHECK(load_always_allowed(ws).empty());
    REQUIRE(save_always_allowed(ws, {"write_file", "run_command"}));
    const auto loaded = load_always_allowed(ws);
    REQUIRE(loaded.size() == 2);
    CHECK(loaded[1] == "run_command");

    CHECK(remove_conversation_id("a\nb\nc\n", "b") == "a\nc\n");
    CHECK(remove_conversation_id("a\n", "x") == "a\n");
    CHECK(remove_conversation_id("", "x").empty());
}
