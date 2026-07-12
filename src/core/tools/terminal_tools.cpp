#include "../utils/utils.hpp"

#include <agent-cpp/agent.hpp>
#include <cstdio>
#include <simdjson.h>
#include <string>

#ifndef _WIN32
#include <sys/wait.h>
#endif

namespace agent::tools {

static Result<std::string> run_command_cb(std::string_view args, void *ud) {
    const std::string &ws = *static_cast<const std::string *>(ud);

    simdjson::ondemand::parser parser;
    simdjson::padded_string json(args.data(), args.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(json).get(doc))
        return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    simdjson::ondemand::object obj;
    if (doc.get_object().get(obj))
        return fail<std::string>(ErrorCode::ParseError, "Expected object");

    std::string_view command_sv;
    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (k.error())
            continue;
        if (k.value() == "command")
            field.value().get_string().get(command_sv);
    }
    if (command_sv.empty())
        return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'command'");

#ifdef _WIN32
    std::string full_cmd = "cd /d \"" + ws + "\" && (" + std::string(command_sv) + ") 2>&1";
    FILE *pipe = _popen(full_cmd.c_str(), "r");
#else
    std::string full_cmd = "cd '" + ws + "' && (" + std::string(command_sv) + ") 2>&1";
    FILE *pipe = popen(full_cmd.c_str(), "r");
#endif

    if (!pipe)
        return fail<std::string>(ErrorCode::FilesystemError, "Failed to execute command");

    std::string output;
    char buf[256];
    while (fgets(buf, sizeof(buf), pipe)) {
        output += buf;
        if (output.size() > 32 * 1024) {
            output += "\n[output truncated at 32 KiB]";
            while (fgets(buf, sizeof(buf), pipe)) {
            }
            break;
        }
    }

#ifdef _WIN32
    int exit_code = _pclose(pipe);
#else
    int raw = pclose(pipe);
    int exit_code = WIFEXITED(raw) ? WEXITSTATUS(raw) : -1;
#endif

    std::string result = "{\"exit_code\":" + std::to_string(exit_code) + ",\"output\":\"";
    utils::escape_json_string(output, result);
    result += "\"}";
    return ok(std::move(result));
}

Tool get_terminal_tool(const std::string *workspace_dir) {
    void *ud = const_cast<void *>(static_cast<const void *>(workspace_dir));
    return Tool{
        .name = "run_command",
        .description =
            "Run a shell command in the workspace directory. stdout and stderr are merged. "
            "Returns JSON: {exit_code, output}. Absolute paths in the command can access the broader filesystem.",
        .parameter_schema =
            R"({"type":"object","properties":{"command":{"type":"string","description":"shell command to execute"}},"required":["command"]})",
        .callback = run_command_cb,
        .user_data = ud,
    };
}

} // namespace agent::tools
