#include "../utils/utils.hpp"
#include "hal/process.hpp"

#include <agent-cpp/agent.hpp>
#include <simdjson.h>
#include <string>

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
    int64_t timeout_seconds = 30;

    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (k.error())
            continue;
        if (k.value() == "command") {
            (void)field.value().get_string().get(command_sv);
        } else if (k.value() == "timeout_seconds") {
            int64_t v = 0;
            if (!field.value().get_int64().get(v))
                timeout_seconds = v;
        }
    }
    if (command_sv.empty())
        return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'command'");

    auto proc = hal::run_shell_command(ws, command_sv, static_cast<int>(timeout_seconds));
    if (!proc.ok)
        return fail<std::string>(proc.error.code, proc.error.message);

    std::string result = "{\"exit_code\":" + std::to_string(proc.value.exit_code) +
                         ",\"timed_out\":" + (proc.value.timed_out ? "true" : "false") +
                         ",\"output\":\"";
    utils::escape_json_string(proc.value.output, result);
    result += "\"}";
    return ok(std::move(result));
}

Tool get_terminal_tool(const std::string *workspace_dir) {
    return Tool{
        .name = "run_command",
        .description =
            "Run a shell command in the workspace directory. stdout and stderr are merged. "
            "Returns JSON: {exit_code, timed_out, output}.\n"
            "Parameters: command (string, required), timeout_seconds (integer, default 30, "
            "0=no timeout).\n"
            "Timeout is enforced on all platforms (Windows CreateProcess wait, POSIX timeout/"
            "wait). Absolute paths in commands can still access outside the workspace.",
        .parameter_schema =
            R"({"type":"object","properties":{"command":{"type":"string"},"timeout_seconds":{"type":"integer","description":"max seconds, default 30, 0 = no timeout"}},"required":["command"]})",
        .callback = run_command_cb,
        .user_data = const_cast<void *>(static_cast<const void *>(workspace_dir)),
    };
}

} // namespace agent::tools
