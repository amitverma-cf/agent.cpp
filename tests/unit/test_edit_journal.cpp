#include "edit_journal.hpp"

#include <catch_amalgamated.hpp>
#include <filesystem>
#include <fstream>
#include <iterator>

using namespace agent::cli;
namespace fs = std::filesystem;

namespace {
std::string read_all(const fs::path &p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}
void write_all(const fs::path &p, const std::string &text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << text;
}
std::string fresh_dir(const char *name) {
    const fs::path dir = fs::temp_directory_path() / name;
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir.string();
}
} // namespace

TEST_CASE("edit journal: unified diff", "[cli][journal]") {
    CHECK(unified_diff("a\nb\n", "a\nb\n", "f.txt").empty());
    const std::string d = unified_diff("one\ntwo\nthree\nfour\nfive\n", "one\ntwo\nTHREE\nfour\nfive\nsix\n", "f.txt");
    CHECK(d.find("--- a/f.txt") != std::string::npos);
    CHECK(d.find("+++ b/f.txt") != std::string::npos);
    CHECK(d.find("-three") != std::string::npos);
    CHECK(d.find("+THREE") != std::string::npos);
    CHECK(d.find("+six") != std::string::npos);
    CHECK(d.find("@@ -1,5 +1,6 @@") != std::string::npos);
    CHECK(d.find("\n one\n") != std::string::npos); // unchanged lines are context, with a leading space

    // Two changes far apart are two hunks.
    std::string a, b;
    for (int i = 0; i < 40; ++i) a += "line" + std::to_string(i) + "\n";
    b = a;
    b.replace(b.find("line2\n"), 6, "LINE2\n");
    b.replace(b.find("line35\n"), 7, "LINE35\n");
    const std::string two = unified_diff(a, b, "g.txt");
    size_t hunks = 0;
    for (size_t at = two.find("@@"); at != std::string::npos; at = two.find("@@", two.find('\n', at))) {
        ++hunks;
        at = two.find('\n', at);
        if (at == std::string::npos) break;
    }
    CHECK(hunks == 2);
}

TEST_CASE("edit journal: the files a tool call changes", "[cli][journal]") {
    CHECK(files_changed_by_tool("write_file", R"({"path":"a.txt","content":"x"})") == std::vector<std::string>{"a.txt"});
    CHECK(files_changed_by_tool("move_path", R"({"from":"a","to":"b"})") == std::vector<std::string>{"a", "b"});
    CHECK(files_changed_by_tool("read_file", R"({"path":"a"})").empty());
    CHECK(files_changed_by_tool("write_file", "not json").empty());
}

TEST_CASE("edit journal: diff and undo", "[cli][journal]") {
    const std::string ws = fresh_dir("agent_journal_ws");
    write_all(fs::path(ws) / "keep.txt", "original\n");
    EditJournal journal(ws);
    CHECK(journal.undo().empty());
    CHECK(journal.diff() == "no changes to files in this session");

    // A tool call overwrites a file and creates another.
    journal.before_tool("write_file", R"({"path":"keep.txt","content":"changed\n"})");
    write_all(fs::path(ws) / "keep.txt", "changed\n");
    journal.before_tool("write_file", R"({"path":"new.txt","content":"fresh\n"})");
    write_all(fs::path(ws) / "new.txt", "fresh\n");

    const std::string diff = journal.diff();
    CHECK(diff.find("-original") != std::string::npos);
    CHECK(diff.find("+changed") != std::string::npos);
    CHECK(diff.find("new file: new.txt") != std::string::npos);

    // The last call is undone first: the new file goes, then the old content comes back.
    CHECK(journal.undo().find("removed new.txt") != std::string::npos);
    CHECK_FALSE(fs::exists(fs::path(ws) / "new.txt"));
    CHECK(journal.undo().find("restored keep.txt") != std::string::npos);
    CHECK(read_all(fs::path(ws) / "keep.txt") == "original\n");
    CHECK(journal.undo().empty());
    CHECK(journal.diff() == "no changes to files in this session");

    // A moved file is put back in one step.
    write_all(fs::path(ws) / "src.txt", "data\n");
    journal.before_tool("move_path", R"({"from":"src.txt","to":"dst.txt"})");
    fs::rename(fs::path(ws) / "src.txt", fs::path(ws) / "dst.txt");
    const std::string back = journal.undo();
    CHECK(back.find("restored src.txt") != std::string::npos);
    CHECK(read_all(fs::path(ws) / "src.txt") == "data\n");
    CHECK_FALSE(fs::exists(fs::path(ws) / "dst.txt"));
}
