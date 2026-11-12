#pragma once

// Secret values (API keys, tokens) for MCP servers, tools and plugins.
//
// Rules this module enforces:
//   * Values live in the operating system's credential store (Windows Credential Manager, macOS Keychain, the
//     Linux Secret Service through `secret-tool`). A machine without one falls back to an encrypted file that
//     needs a passphrase. Values are never written to config files, history, logs or tool output.
//   * A secret is bound to the things allowed to use it (an MCP server, a tool, a plugin, or a group of them). The
//     binding names the consumer's identity: a hash of what it runs. A changed command is a different consumer and
//     is refused until the user approves it again.
//   * The model never sees a secret and has no tool that reads or manages them. Only the host resolves a secret,
//     at the moment it starts the bound process, and puts it in that process's environment.
//
// The names and bindings are not secret and live in <home>/secrets.json; values live only in the store.

#include <agent-cpp/agent.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace agent::cli {

// A secret's name: 1-64 characters from [A-Za-z0-9_.-]. Keeps names safe in OS commands and file names.
bool valid_secret_name(std::string_view name);

// The stores a value can live in. `available` is a cheap probe; `name` is shown to the user.
struct SecretBackend {
    const char *name;
    void *state; // backend-private data, passed to every call (the encrypted file's path and passphrase)
    bool (*available)(void *state);
    Result<void> (*set)(void *state, std::string_view name, std::string_view value);
    Result<std::string> (*get)(void *state, std::string_view name);
    Result<void> (*remove)(void *state, std::string_view name);
};

// The best backend on this machine: the OS store when it works, otherwise the encrypted file.
const SecretBackend &default_secret_backend(const std::string &home_root);
// The encrypted-file backend for a given file (needs $AGENT_SECRETS_PASSPHRASE or a terminal prompt), also
// used directly by tests.
SecretBackend encrypted_file_backend(const std::string &path);

// What a consumer is: "mcp:github", "tool:web_search", "group:research". `identity` is a hash of what it runs.
struct SecretBinding {
    std::string consumer;
    std::string identity;
};

struct SecretEntry {
    std::string name;
    std::vector<SecretBinding> bindings;
};

// <home>/secrets.json: names and bindings only.
Result<std::vector<SecretEntry>> load_secret_index(const std::string &path);
Result<void> save_secret_index(const std::string &path, const std::vector<SecretEntry> &entries);

// sha256 (lowercase hex) over the parts, separated so ("a","bc") differs from ("ab","c").
std::string consumer_identity(const std::vector<std::string> &parts);

enum class SecretAccess { Allowed, UnknownSecret, NotBound, IdentityChanged };
// May `consumer` (with `identity`) use the secret called `name`?
SecretAccess check_secret_access(const std::vector<SecretEntry> &index, std::string_view name, std::string_view consumer,
                                 std::string_view identity);

// Replaces every ${secret:NAME} in `text` with the value, after check_secret_access. Fails (and leaves `out`
// empty) on the first name that is unknown, not bound to this consumer, or bound to a different identity.
// Each value inserted is also appended to *values when given, so the caller can redact it from output.
Result<std::string> resolve_secret_placeholders(std::string_view text, const SecretBackend &backend, const std::vector<SecretEntry> &index,
                                                std::string_view consumer, std::string_view identity,
                                                std::vector<std::string> *values = nullptr);

// The store and the index of names and bindings, loaded once for a session.
struct SecretContext {
    SecretBackend backend{};
    std::vector<SecretEntry> index;
};
Result<std::shared_ptr<SecretContext>> load_secret_context(const std::string &home_root, const std::string &index_path);

// Config::secret_resolver for the CLI: user_data is a SecretContext. The identity is the hash of command + args.
Result<std::string> cli_resolve_secrets(std::string_view text, const SecretRequest &who, std::vector<std::string> &secret_values,
                                        void *user_data);

// Replaces every known secret value found in `text` with "***". Defence in depth for tool output and logs.
std::string redact_secrets(std::string text, const std::vector<std::string> &values);

} // namespace agent::cli
