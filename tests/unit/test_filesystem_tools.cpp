#include <catch_amalgamated.hpp>

#include <agent-cpp/agent.hpp>
#include <filesystem>
#include <string>

namespace {

// Each TEST_CASE gets its own workspace subdirectory since these tests do real file I/O.
std::string make_workspace(const char *name) {
    std::string dir = "/tmp/agent_test_fs_" + std::string(name);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    return dir;
}

const agent::Tool *find_tool(const agent::Session &session, std::string_view name) {
    for (const auto &t : session.config.tools)
        if (t.name == name)
            return &t;
    return nullptr;
}

} // namespace

TEST_CASE("write_file then read_file round-trips content", "[filesystem_tools]") {
    std::string ws = make_workspace("rw");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *write_tool = find_tool(session, "write_file");
    const auto *read_tool = find_tool(session, "read_file");
    REQUIRE(write_tool != nullptr);
    REQUIRE(read_tool != nullptr);

    auto write_res =
        write_tool->callback(R"({"path":"note.txt","content":"hello world"})", write_tool->user_data);
    REQUIRE(write_res.ok);
    REQUIRE(write_res.value.find("wrote 11 bytes") != std::string::npos);

    auto read_res = read_tool->callback(R"({"path":"note.txt"})", read_tool->user_data);
    REQUIRE(read_res.ok);
    REQUIRE(read_res.value == "hello world");
}

TEST_CASE("read_file truncates at max_bytes", "[filesystem_tools]") {
    std::string ws = make_workspace("truncate");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *write_tool = find_tool(session, "write_file");
    const auto *read_tool = find_tool(session, "read_file");

    auto write_res =
        write_tool->callback(R"({"path":"big.txt","content":"0123456789"})", write_tool->user_data);
    REQUIRE(write_res.ok);

    auto read_res = read_tool->callback(R"({"path":"big.txt","max_bytes":4})", read_tool->user_data);
    REQUIRE(read_res.ok);
    REQUIRE(read_res.value.find("0123") == 0);
    REQUIRE(read_res.value.find("[truncated at 4 bytes]") != std::string::npos);
}

TEST_CASE("read_file on missing file returns KeyNotFound", "[filesystem_tools]") {
    std::string ws = make_workspace("missing");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *read_tool = find_tool(session, "read_file");
    auto read_res = read_tool->callback(R"({"path":"nope.txt"})", read_tool->user_data);
    REQUIRE_FALSE(read_res.ok);
}

TEST_CASE("append_file appends to an existing and a new file", "[filesystem_tools]") {
    std::string ws = make_workspace("append");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *append_tool = find_tool(session, "append_file");
    const auto *read_tool = find_tool(session, "read_file");

    auto a1 = append_tool->callback(R"({"path":"log.txt","content":"line1\n"})", append_tool->user_data);
    REQUIRE(a1.ok);
    auto a2 = append_tool->callback(R"({"path":"log.txt","content":"line2\n"})", append_tool->user_data);
    REQUIRE(a2.ok);

    auto read_res = read_tool->callback(R"({"path":"log.txt"})", read_tool->user_data);
    REQUIRE(read_res.ok);
    REQUIRE(read_res.value == "line1\nline2\n");
}

TEST_CASE("list_dir lists files and directories with metadata", "[filesystem_tools]") {
    std::string ws = make_workspace("list");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *write_tool = find_tool(session, "write_file");
    const auto *create_dir_tool = find_tool(session, "create_dir");
    const auto *list_tool = find_tool(session, "list_dir");

    REQUIRE(write_tool->callback(R"({"path":"a.txt","content":"x"})", write_tool->user_data).ok);
    REQUIRE(create_dir_tool->callback(R"({"path":"subdir"})", create_dir_tool->user_data).ok);

    auto list_res = list_tool->callback("", list_tool->user_data);
    REQUIRE(list_res.ok);
    REQUIRE(list_res.value.find("\"a.txt\"") != std::string::npos);
    REQUIRE(list_res.value.find("\"type\":\"file\"") != std::string::npos);
    REQUIRE(list_res.value.find("\"subdir\"") != std::string::npos);
    REQUIRE(list_res.value.find("\"type\":\"dir\"") != std::string::npos);
}

TEST_CASE("create_dir creates nested directories", "[filesystem_tools]") {
    std::string ws = make_workspace("mkdir");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *create_dir_tool = find_tool(session, "create_dir");
    auto res = create_dir_tool->callback(R"({"path":"a/b/c"})", create_dir_tool->user_data);
    REQUIRE(res.ok);
    REQUIRE(std::filesystem::is_directory(ws + "/a/b/c"));
}

TEST_CASE("delete_path removes files and refuses the workspace root", "[filesystem_tools]") {
    std::string ws = make_workspace("delete");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *write_tool = find_tool(session, "write_file");
    const auto *delete_tool = find_tool(session, "delete_path");

    REQUIRE(write_tool->callback(R"({"path":"gone.txt","content":"x"})", write_tool->user_data).ok);
    auto del_res = delete_tool->callback(R"({"path":"gone.txt"})", delete_tool->user_data);
    REQUIRE(del_res.ok);
    REQUIRE_FALSE(std::filesystem::exists(ws + "/gone.txt"));

    auto root_res = delete_tool->callback(R"({"path":"."})", delete_tool->user_data);
    REQUIRE_FALSE(root_res.ok);
    REQUIRE(root_res.error.code == agent::ErrorCode::SandboxViolation);
}

TEST_CASE("move_path renames a file", "[filesystem_tools]") {
    std::string ws = make_workspace("move");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *write_tool = find_tool(session, "write_file");
    const auto *move_tool = find_tool(session, "move_path");
    const auto *read_tool = find_tool(session, "read_file");

    REQUIRE(write_tool->callback(R"({"path":"src.txt","content":"payload"})", write_tool->user_data).ok);
    auto move_res = move_tool->callback(R"({"from":"src.txt","to":"dst/dst.txt"})", move_tool->user_data);
    REQUIRE(move_res.ok);

    auto read_res = read_tool->callback(R"({"path":"dst/dst.txt"})", read_tool->user_data);
    REQUIRE(read_res.ok);
    REQUIRE(read_res.value == "payload");
}

TEST_CASE("file_info reports existence, type, and size", "[filesystem_tools]") {
    std::string ws = make_workspace("info");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *write_tool = find_tool(session, "write_file");
    const auto *info_tool = find_tool(session, "file_info");

    REQUIRE(write_tool->callback(R"({"path":"f.bin","content":"abcd"})", write_tool->user_data).ok);

    auto info_res = info_tool->callback(R"({"path":"f.bin"})", info_tool->user_data);
    REQUIRE(info_res.ok);
    REQUIRE(info_res.value.find("\"exists\":true") != std::string::npos);
    REQUIRE(info_res.value.find("\"type\":\"file\"") != std::string::npos);
    REQUIRE(info_res.value.find("\"size\":4") != std::string::npos);

    auto missing_res = info_tool->callback(R"({"path":"nope.bin"})", info_tool->user_data);
    REQUIRE(missing_res.ok);
    REQUIRE(missing_res.value == R"({"exists":false})");
}

TEST_CASE("Filesystem tools reject paths that escape the workspace sandbox", "[filesystem_tools][sandbox]") {
    std::string ws = make_workspace("sandbox");
    auto session_result = agent::init({.provider = agent::AiProvider::Mock, .workspace_dir = ws});
    REQUIRE(session_result.ok);
    agent::Session session = std::move(session_result.value);

    const auto *read_tool = find_tool(session, "read_file");
    auto res = read_tool->callback(R"({"path":"../../etc/passwd"})", read_tool->user_data);
    REQUIRE_FALSE(res.ok);
    REQUIRE(res.error.code == agent::ErrorCode::SandboxViolation);
}
