#include "path_gate.hpp"
#include <simdjson.h>
#include "../utils/utils.hpp"
#include "hal/process.hpp"

#include <agent-cpp/agent.hpp>
#include <filesystem>
#include <memory>
#include <simdjson.h>
#include <string>

namespace agent::tools {

namespace {

Result<std::string> run_command(const std::string &ws, std::string_view args, const SandboxPolicy *sandbox,
                                const volatile int32_t *cancel) {
    simdjson::ondemand::parser parser;
    simdjson::padded_string json(args.data(), args.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(json).get(doc)) return fail<std::string>(ErrorCode::ParseError, "Invalid JSON");

    simdjson::ondemand::object obj;
    if (doc.get_object().get(obj)) return fail<std::string>(ErrorCode::ParseError, "Expected object");

    std::string_view command_sv;
    int64_t timeout_seconds = 30;

    for (auto field : obj) {
        auto k = field.unescaped_key();
        if (k.error()) continue;
        if (k.value() == "command") {
            { if (field.value().get_string().get(command_sv)) command_sv = {}; }
        } else if (k.value() == "timeout_seconds") {
            int64_t v = 0;
            if (!field.value().get_int64().get(v)) timeout_seconds = v;
        }
    }
    if (command_sv.empty()) return fail<std::string>(ErrorCode::InvalidConfig, "Missing 'command'");
    // Negative = default; at most a day (the int cast below must not overflow).
    if (timeout_seconds < 0) timeout_seconds = 30;
    if (timeout_seconds > 86400) timeout_seconds = 86400;

    auto proc = hal::run_shell_command(ws, command_sv, static_cast<int>(timeout_seconds), 32 * 1024, sandbox, cancel);
    if (!proc.ok) return fail<std::string>(proc.error.code, proc.error.message);

    std::string result = "{\"exit_code\":" + std::to_string(proc.value.exit_code) +
                         ",\"timed_out\":" + (proc.value.timed_out ? "true" : "false") +
                         ",\"cancelled\":" + (proc.value.cancelled ? "true" : "false") + ",\"output\":\"";
    // The output cap can cut a UTF-8 sequence, and programs print arbitrary bytes: keep the JSON valid text.
    utils::sanitize_utf8(proc.value.output);
    utils::escape_json_string(proc.value.output, result);
    result += "\"";
    if (!proc.value.sandbox_warnings.empty()) {
        // Tell the caller (and the model) that the sandbox is weaker than configured on this machine.
        std::string joined;
        for (const auto &w : proc.value.sandbox_warnings) joined += (joined.empty() ? "" : " ") + w;
        result += ",\"sandbox_warning\":\"";
        utils::escape_json_string(joined, result);
        result += "\"";
    }
    result += "}";
    return ok(std::move(result));
}

Result<std::string> run_command_cb(std::string_view args, void *ud) {
    return run_command(*static_cast<const std::string *>(ud), args, nullptr, nullptr);
}

// The default tool set's run_command: the workspace is the only writable (and, outside the system
// dirs, the only readable) place when the path sandbox is on.
Result<std::string> run_command_sandboxed_cb(std::string_view args, void *ud) {
    const auto &ctx = *static_cast<const WorkspaceContext *>(ud);
    if (!ctx.workspace_dir) return fail<std::string>(ErrorCode::InvalidConfig, "workspace_dir is null.");
    const std::string &ws = *ctx.workspace_dir;
    if (!ctx.command_sandbox.enabled) return run_command(ws, args, nullptr, ctx.cancel);
    SandboxPolicy policy = ctx.command_sandbox;
    policy.filesystem = policy.filesystem && ctx.sandbox;
    policy.windows_low_integrity = true; // Windows: the command writes only in the workspace
    if (!ws.empty()) {
        std::error_code ec;
        auto abs = std::filesystem::absolute(ws, ec);
        policy.read_write_paths.push_back(ec ? ws : abs.string());
    }
    // Paths outside the workspace: grants first, then the gate for each other absolute path the command line names.
    apply_grants_to_policy(ctx, "run_command", policy);
    {
        simdjson::ondemand::parser parser;
        simdjson::padded_string json(args.data(), args.size());
        simdjson::ondemand::document doc;
        std::string_view command;
        if (!parser.iterate(json).get(doc) && !doc["command"].get_string().get(command) && !ws.empty()) {
            const bool writes = command_looks_like_write(command);
            for (const std::filesystem::path &p : external_paths_in_command(command, ws)) {
                if (path_access_allowed(ctx, "run_command", p, writes)) (writes ? policy.read_write_paths : policy.read_only_paths).push_back(p.string());
            }
        }
    }
    return run_command(ws, args, &policy, ctx.cancel);
}

constexpr const char *kDescription = "Run a shell command in the workspace directory. stdout and stderr are merged. "
                                     "Returns JSON: {exit_code, timed_out, cancelled, output}.\n"
                                     "Parameters: command (string, required), timeout_seconds (integer, default 30, "
                                     "0=no timeout).\n"
                                     "The timeout is enforced on all platforms and stops the command's child processes "
                                     "too. If the turn is cancelled the command is killed the same way and cancelled "
                                     "is true; output holds what it printed so far.";

constexpr const char *kSchema =
    R"({"type":"object","properties":{"command":{"type":"string"},"timeout_seconds":{"type":"integer","description":"max seconds, default 30, 0 = no timeout"}},"required":["command"]})";

} // namespace

Tool get_terminal_tool(const std::string *workspace_dir) {
    return Tool{
        .name = "run_command",
        .description = kDescription,
        .parameter_schema = kSchema,
        .callback = run_command_cb,
        .user_data = const_cast<void *>(static_cast<const void *>(workspace_dir)),
    };
}

Tool get_terminal_tool(std::shared_ptr<WorkspaceContext> ctx) {
    return Tool{
        .name = "run_command",
        .description = kDescription,
        .parameter_schema = kSchema,
        .callback = run_command_sandboxed_cb,
        .user_data = ctx.get(),
    };
}

} // namespace agent::tools
