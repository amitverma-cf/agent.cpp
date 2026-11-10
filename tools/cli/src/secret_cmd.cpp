// agent secret: keys for MCP servers, tools and plugins. The value goes straight into the OS credential store;
// nothing here ever prints it, and the model has no tool that reads or manages secrets.

#include "commands.hpp"
#include "secrets.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

std::string index_path(const Context &ctx) { return (fs::path(ctx.has_home ? ctx.home.root : ctx.config_dir) / "secrets.json").string(); }

SecretEntry *find_entry(std::vector<SecretEntry> &index, std::string_view name) {
    auto it = std::find_if(index.begin(), index.end(), [&](const SecretEntry &e) { return e.name == name; });
    return it == index.end() ? nullptr : &*it;
}

// A consumer is "mcp:NAME", "tool:NAME", "plugin:NAME" or "group:NAME".
bool valid_consumer(std::string_view c) {
    for (std::string_view kind : {"mcp:", "tool:", "plugin:", "group:"})
        if (c.substr(0, kind.size()) == kind) return valid_secret_name(c.substr(kind.size()));
    return false;
}

} // namespace

int cmd_secret(const CommandLine &cl, Context &ctx) {
    const SecretBackend &backend = default_secret_backend(ctx.has_home ? ctx.home.root : ctx.config_dir);
    const std::string path = index_path(ctx);
    auto loaded = load_secret_index(path);
    if (!loaded.ok) return report_error(loaded.error);
    std::vector<SecretEntry> index = std::move(loaded.value);

    if (cl.subcommand == "ls") {
        std::printf("store: %s\n", backend.name);
        if (index.empty()) std::printf("no secrets yet: `agent secret set NAME`\n");
        for (const SecretEntry &e : index) {
            std::printf("%-24s", e.name.c_str());
            if (e.bindings.empty()) std::printf("  not bound to anything (nothing can use it)\n");
            else {
                std::printf("  usable by:");
                for (const SecretBinding &b : e.bindings) std::printf(" %s", b.consumer.c_str());
                std::printf("\n");
            }
        }
        return 0; // values are never listed
    }

    const std::string &name = cl.positionals[0];
    if (!valid_secret_name(name)) return report_error(Error{ErrorCode::InvalidConfig, "secret name must be 1-64 characters from A-Z a-z 0-9 _ . -"});

    if (cl.subcommand == "set") {
        std::string value;
        if (cl.has("stdin")) {
            std::getline(std::cin, value);
            while (!value.empty() && (value.back() == '\r' || value.back() == '\n')) value.pop_back();
        } else if (!read_hidden_line(("Value for " + name + " (not shown): ").c_str(), value)) {
            return report_error(Error{ErrorCode::InvalidConfig, "no terminal to read the value from: pipe it in with --stdin"});
        }
        if (value.empty()) return report_error(Error{ErrorCode::InvalidConfig, "empty value: nothing stored"});
        if (auto r = backend.set(backend.state, name, value); !r.ok) return report_error(r.error);
        std::fill(value.begin(), value.end(), '\0');
        if (!find_entry(index, name)) index.push_back(SecretEntry{name, {}});
        if (auto r = save_secret_index(path, index); !r.ok) return report_error(r.error);
        std::printf("stored '%s' in %s.\nIt is not usable yet: allow the server or tool that needs it, e.g. `agent secret bind %s mcp:%s`.\n", name.c_str(), backend.name, name.c_str(),
                    name.c_str());
        return 0;
    }

    if (cl.subcommand == "rm") {
        if (auto r = backend.remove(backend.state, name); !r.ok) return report_error(r.error);
        index.erase(std::remove_if(index.begin(), index.end(), [&](const SecretEntry &e) { return e.name == name; }), index.end());
        if (auto r = save_secret_index(path, index); !r.ok) return report_error(r.error);
        std::printf("removed '%s'\n", name.c_str());
        return 0;
    }

    const std::string &consumer = cl.positionals[1];
    if (!valid_consumer(consumer)) return report_error(Error{ErrorCode::InvalidConfig, "consumer must look like mcp:NAME, tool:NAME, plugin:NAME or group:NAME"});
    SecretEntry *entry = find_entry(index, name);
    if (!entry) return report_error(Error{ErrorCode::KeyNotFound, "no secret named '" + name + "': add it with `agent secret set " + name + "`"});

    if (cl.subcommand == "bind") {
        // The identity pins what the consumer runs. Until MCP servers and plugins report their command line, a consumer is
        // identified by its declared name; once they do, `bind` records the hash of that command and a changed command is refused.
        const std::string identity = consumer_identity({"declared", consumer});
        auto existing = std::find_if(entry->bindings.begin(), entry->bindings.end(), [&](const SecretBinding &b) { return b.consumer == consumer; });
        if (existing != entry->bindings.end()) existing->identity = identity;
        else entry->bindings.push_back(SecretBinding{consumer, identity});
        if (auto r = save_secret_index(path, index); !r.ok) return report_error(r.error);
        std::printf("'%s' can now be used by %s (and only by it)\n", name.c_str(), consumer.c_str());
        return 0;
    }

    // unbind
    entry->bindings.erase(std::remove_if(entry->bindings.begin(), entry->bindings.end(), [&](const SecretBinding &b) { return b.consumer == consumer; }), entry->bindings.end());
    if (auto r = save_secret_index(path, index); !r.ok) return report_error(r.error);
    std::printf("%s can no longer use '%s'\n", consumer.c_str(), name.c_str());
    return 0;
}

} // namespace agent::cli
