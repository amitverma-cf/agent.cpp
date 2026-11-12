// agent setup / engine / model / run, plus helpers shared with chat.cpp.

#include "commands.hpp"

#include "doctor.hpp"
#include "engine_release.hpp"
#include "runtimes.hpp"
#include "status_view.hpp"

#include <agent-cpp/engine_abi.h>
#include <agent-cpp/hub.hpp>
#include <agent-cpp/store_sqlite.hpp>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace agent::cli {

namespace fs = std::filesystem;

namespace {

std::string join(std::span<const std::string> items, std::string_view sep) {
    std::string out;
    for (size_t i = 0; i < items.size(); ++i) out += (i ? std::string(sep) : "") + items[i];
    return out;
}

std::string absolute_path(std::string_view p) {
    std::error_code ec;
    fs::path abs = fs::absolute(fs::path(p), ec);
    return ec ? std::string(p) : abs.lexically_normal().string();
}

// Prompt with a default; returns the default on an empty answer, EOF or non-interactive use.
std::string ask_value(bool interactive, std::string_view question, std::string_view current) {
    if (!interactive) return std::string(current);
    std::fprintf(stderr, "%.*s [%.*s]: ", static_cast<int>(question.size()), question.data(), static_cast<int>(current.size()),
                 current.data());
    std::fflush(stderr);
    std::string line;
    if (read_line(stdin, line) != ReadStatus::Line) {
        std::fprintf(stderr, "\n");
        return std::string(current);
    }
    while (!line.empty() && (line.back() == ' ' || line.back() == '\t')) line.pop_back();
    size_t start = line.find_first_not_of(" \t");
    if (start == std::string::npos) return std::string(current);
    return line.substr(start);
}

Result<LoadedEngine> load_plugin_info(const std::string &path) {
    auto name = agent::load_engine_plugin(path);
    if (!name.ok) return fail<LoadedEngine>(name.error.code, name.error.message);
    const agent_engine_v1 *e = agent::find_engine(name.value);
    LoadedEngine out;
    out.name = name.value;
    out.path = path;
    if (e) {
        out.version = e->version ? e->version : "";
        out.caps = e->caps;
    }
    return ok(std::move(out));
}

// -------------------------- model pull progress / Ctrl-C --------------------------

std::atomic<bool> g_pull_cancel{false};

void on_pull_signal(int) { g_pull_cancel.store(true); }

struct PullProgress {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last{};
    uint64_t first_bytes = UINT64_MAX; // bytes already on disk when the transfer (re)started
};

bool print_pull_progress(uint64_t done, uint64_t total, void *user_data) {
    auto *p = static_cast<PullProgress *>(user_data);
    const auto now = std::chrono::steady_clock::now();
    if (p->first_bytes == UINT64_MAX) p->first_bytes = done;
    if (done != total && now - p->last < std::chrono::milliseconds(100)) return !g_pull_cancel.load();
    p->last = now;
    const double secs = std::chrono::duration<double>(now - p->start).count();
    const double rate = secs > 0.0 ? static_cast<double>(done - p->first_bytes) / secs / (1024.0 * 1024.0) : 0.0;
    std::fprintf(stderr, "\r  %s  %6.1f MiB/s ", format_progress(done, total).c_str(), rate);
    std::fflush(stderr);
    return !g_pull_cancel.load();
}

} // namespace

int report_error(const Error &e) {
    std::fprintf(stderr, "error: %s\n", e.message.c_str());
    return 1;
}

bool ask_yes_no(std::string_view question) {
    std::fprintf(stderr, "%.*s [y/N] ", static_cast<int>(question.size()), question.data());
    std::fflush(stderr);
    std::string line;
    if (read_line(stdin, line) != ReadStatus::Line) {
        std::fprintf(stderr, "\n");
        return false;
    }
    const size_t i = line.find_first_not_of(" \t");
    return i != std::string::npos && (line[i] == 'y' || line[i] == 'Y');
}

std::string models_dir(const Context &ctx) {
    if (!ctx.config.models_dir.empty()) return ctx.config.models_dir;
    return ctx.has_home ? ctx.home.models : agent::hub::default_models_dir();
}

std::string catalog_path(const Context &ctx) { return (fs::path(models_dir(ctx)) / "models.json").string(); }

std::string workspace_dir(const Context &ctx, std::string_view override_dir) {
    if (!override_dir.empty()) return std::string(override_dir);
    if (!ctx.config.workspace.empty()) return ctx.config.workspace;
    return ctx.has_home ? (fs::path(ctx.home.workspaces) / "default").string() : default_workspace_dir();
}

std::vector<ModelRef> catalog_refs(const Context &ctx) {
    std::vector<ModelRef> out;
    auto cat = agent::hub::load_catalog(catalog_path(ctx));
    if (!cat.ok) {
        std::fprintf(stderr, "warning: %s\n", cat.error.message.c_str());
        return out;
    }
    for (const auto &e : cat.value) out.push_back(ModelRef{e.name, e.path});
    return out;
}

std::string catalog_engine_options(const Context &ctx, std::string_view model_path, std::string engine_options) {
    auto cat = agent::hub::load_catalog(catalog_path(ctx));
    if (!cat.ok) return engine_options;
    for (const auto &e : cat.value)
        if (e.path == model_path && !e.mmproj.empty()) return agent::hub::with_mmproj_option(engine_options, e.mmproj);
    return engine_options;
}

Result<LoadedEngine> load_engine_for(const Context &ctx, std::string_view engine_flag, std::string_view model_path) {
    auto engines = load_engines(ctx.engines_path);
    if (!engines.ok) return fail<LoadedEngine>(engines.error.code, engines.error.message);
    auto ref = resolve_engine(engine_flag, model_path, engines.value, file_exists);
    if (!ref.ok) return fail<LoadedEngine>(ref.error.code, ref.error.message);

    std::string path = ref.value.path;
    if (path.empty()) {
        const auto dirs = engine_search_dirs();
        path = find_plugin(ref.value.name, dirs);
        if (path.empty())
            return fail<LoadedEngine>(ErrorCode::UnsupportedProvider,
                                      "engine '" + ref.value.name + "' is not installed: it is not in " + ctx.engines_path + " and " +
                                          plugin_file_name(ref.value.name) +
                                          " is not next to the agent binary. Run `agent setup` or `agent engine add <plugin>`.");
    } else if (!file_exists(path)) {
        return fail<LoadedEngine>(ErrorCode::FilesystemError, "engines.json lists '" + ref.value.name + "' at " + path +
                                                                  ", but the file is missing. Fix it with `agent engine add <plugin>`.");
    }
    auto loaded = load_plugin_info(path);
    if (!loaded.ok) return loaded;
    if (!ref.value.name.empty() && loaded.value.name != ref.value.name)
        return fail<LoadedEngine>(ErrorCode::InvalidConfig,
                                  path + " is the '" + loaded.value.name + "' engine, not '" + ref.value.name + "'");
    loaded.value.backend = ctx.config.backend;
    if (const EngineEntry *entry = find_engine_entry(engines.value, loaded.value.name)) {
        loaded.value.isolation = entry->isolation;
        loaded.value.backend = entry->backend;
    }
    return loaded;
}

// -------------------------- agent setup --------------------------

namespace {

// The models this machine can run: registry entries whose engine is installed and whose size fits in RAM with room for
// the KV cache and the rest of the system. Printed as a numbered list; returns the entries in that order.
std::vector<const agent::hub::RegistryEntry *> runnable_models(const Context &ctx, uint64_t ram_bytes) {
    std::vector<const agent::hub::RegistryEntry *> out;
    auto engines = load_engines(ctx.engines_path);
    const uint64_t budget_mib = ram_bytes ? (ram_bytes / (1024ULL * 1024ULL)) * 7 / 10 : 0; // 70% of RAM
    for (const auto &e : agent::hub::model_registry()) {
        if (e.notes.find("NOT runnable") != std::string_view::npos) continue;
        if (!engines.ok || !find_engine_entry(engines.value, std::string(e.engine))) continue;
        if (budget_mib && e.size_mib && static_cast<uint64_t>(e.size_mib) * 13 / 10 > budget_mib)
            continue; // weights + ~30% for KV and buffers
        out.push_back(&e);
    }
    std::stable_sort(out.begin(), out.end(), [](const auto *a, const auto *b) { return a->tested && !b->tested; });
    return out;
}

CommandLine make_command(std::string command, std::string sub, std::vector<std::string> positionals, std::vector<Flag> flags = {}) {
    CommandLine c;
    c.command = std::move(command);
    c.subcommand = std::move(sub);
    c.positionals = std::move(positionals);
    c.flags = std::move(flags);
    return c;
}

// Runs the real command on a pulled model, so setup proves that the engine, the model and the sandbox work together.
bool smoke_test(Context &ctx, std::string_view task, std::string_view engine, const std::string &model, std::string_view engine_options) {
    std::vector<Flag> flags = {{"engine", std::string(engine)}, {"model", model}};
    if (!engine_options.empty()) flags.push_back({"engine-options", std::string(engine_options)});
    if (task == "text_gen") {
        flags.push_back({"text", "Reply with the single word OK."});
        flags.push_back({"max-tokens", "8"});
    } else if (task == "embed") {
        flags.push_back({"text", "hello world"});
    } else if (task == "rerank") {
        flags.push_back({"text", "capital of France"});
        flags.push_back({"doc", "Paris is the capital of France."});
        flags.push_back({"doc", "Bananas are yellow."});
    } else {
        return true; // needs audio / image input: not smoke-tested here
    }
    std::fflush(stdout);
    return cmd_task(make_command("task", "", {std::string(task)}, std::move(flags)), ctx) == 0;
}

void setup_models(const CommandLine &cl, Context &ctx, bool interactive) {
    if (cl.has("no-models")) return;
    std::vector<std::string> chosen; // registry names or owner/repo specs
    for (const Flag &f : cl.flags)
        if (f.name == "model") chosen.push_back(f.value);

    const uint64_t ram = total_ram_bytes();
    if (chosen.empty() && interactive) {
        const auto options = runnable_models(ctx, ram);
        std::printf("\nModels this machine can run (RAM %.1f GiB; nothing is downloaded until you choose):\n",
                    static_cast<double>(ram) / (1024.0 * 1024.0 * 1024.0));
        if (options.empty()) std::printf("  none of the registry models fit with an installed engine; you can still add your own below.\n");
        for (size_t i = 0; i < options.size(); ++i) {
            const auto &e = *options[i];
            std::printf("  %2zu) %-24s %-9.*s %6u MiB  %s%.*s\n", i + 1, std::string(e.alias).c_str(), static_cast<int>(e.task.size()),
                        e.task.data(), e.size_mib, e.tested ? "[tested] " : "", static_cast<int>(e.notes.size()), e.notes.data());
        }
        const std::string answer =
            ask_value(true, "Choose numbers separated by commas, or type owner/repo[:quant] to add your own (Enter = none)", "");
        for (const std::string &token : split_list(answer)) {
            const bool number =
                !token.empty() && std::all_of(token.begin(), token.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
            if (number) {
                const size_t n = static_cast<size_t>(std::atoi(token.c_str()));
                if (n >= 1 && n <= options.size()) chosen.push_back(std::string(options[n - 1]->alias));
                else std::fprintf(stderr, "ignoring %s: not in the list\n", token.c_str());
            } else {
                chosen.push_back(token);
            }
        }
    }
    if (chosen.empty()) {
        std::printf("\nNo models chosen. See what fits with `agent model search`, then `agent model pull <name>`.\n");
        return;
    }

    for (const std::string &name : chosen) {
        std::printf("\n== %s\n", name.c_str());
        if (cmd_model(make_command("model", "pull", {name}), ctx) != 0) {
            std::printf("   download failed; continuing with the next model\n");
            continue;
        }
        // Which task to smoke-test: the registry knows; for a custom repo, a GGUF is tried as a chat model.
        const auto *entry = agent::hub::find_registry_entry(name);
        const std::string task = entry ? std::string(entry->task) : std::string("text_gen");
        const std::string engine = entry ? std::string(entry->engine) : std::string("llama");
        const std::string model_name = entry ? std::string(entry->alias) : std::string();
        std::string model_ref = model_name;
        if (model_ref.empty()) {
            // A custom spec: the catalogue entry is the newest one.
            auto cat = agent::hub::load_catalog(catalog_path(ctx));
            if (cat.ok && !cat.value.empty()) model_ref = cat.value.back().name;
        }
        if (model_ref.empty()) continue;
        std::printf("   smoke test (%s):\n", task.c_str());
        const bool passed = smoke_test(ctx, task, engine, model_ref, entry ? entry->engine_options : std::string_view());
        std::printf("   %s\n",
                    passed ? "smoke test passed" : "smoke test FAILED: the model downloaded but did not run; see the error above");
    }
}

} // namespace

int cmd_init(const CommandLine &cl, Context &ctx) {
    std::error_code ec;
    const std::string dir = cl.positionals.empty() ? std::filesystem::current_path(ec).string() : cl.positionals[0];
    auto made = init_project(dir);
    if (!made.ok) return report_error(made.error);
    ctx.project_root = made.value.root;
    std::printf("project %s\n", made.value.root.c_str());
    std::printf("  %s  skills for this project (committed)\n", made.value.skills.c_str());
    std::printf("  %s  MCP servers for this project (committed; keys are referenced by name, never stored)\n",
                made.value.mcp_config.c_str());
    std::printf("  %s  conversations and memory (private, git-ignored)\n", made.value.memory.c_str());
    std::printf("  %s  private, git-ignored\n", made.value.cache.c_str());
    std::printf("  %s  private, git-ignored\n", made.value.local.c_str());
    return 0;
}

int cmd_setup(const CommandLine &cl, Context &ctx) {
    const bool interactive = stdin_is_tty() && !cl.has("yes");
    const HardwareInfo hw = detect_hardware();

    // Where agent.cpp keeps its files. An explicit old-style config folder skips this.
    {
        const char *env_config = std::getenv("AGENT_CONFIG_DIR");
        if (!cl.has("config-dir") && !(env_config && *env_config)) {
            const std::string suggestion =
                cl.has("home") ? std::string(cl.get("home")) : (ctx.has_home ? ctx.home.root : default_home_suggestion());
            const std::string chosen = absolute_path(
                ask_value(interactive, "Where should agent.cpp keep its files (engines, models, skills, memory)?", suggestion));
            ctx.home = home_layout(chosen);
            ctx.has_home = true;
            ctx.config_dir = ctx.home.root;
            ctx.config_path = ctx.home.config;
            ctx.engines_path = ctx.home.engines_json;
            if (auto r = create_home_layout(ctx.home); !r.ok) return report_error(r.error);
            if (auto r = save_home_pointer(home_pointer_path(), ctx.home.root); !r.ok) return report_error(r.error);
            if (auto loaded = load_config(ctx.config_path); loaded.ok) ctx.config = std::move(loaded.value);
            std::printf("home: %s\n\n", ctx.home.root.c_str());
        }
    }

    std::printf("Hardware\n");
    std::printf("  CPU cores:        %d (logical)\n", hw.cpu_cores);
    std::printf("  nvidia-smi:       %s\n", hw.nvidia_smi ? "found" : "not found");
#if !defined(_WIN32)
    std::printf("  /dev/nvidia0:     %s\n", hw.nvidia_device ? "present" : "absent");
    std::printf("  DRI render nodes: %s\n", hw.dri_nodes.empty() ? "none" : join(hw.dri_nodes, ", ").c_str());
#endif
    const std::string suggested = suggest_backend(hw);
    std::printf("  suggested backend: %s\n\n", suggested.c_str());

    CliConfig cfg = ctx.config;
    cfg.hardware = hw;

    std::string caps_default;
    if (cl.has("caps")) caps_default = std::string(cl.get("caps"));
    else if (!cfg.capabilities.empty()) caps_default = join(cfg.capabilities, ",");
    else
        for (std::string_view c : starter_pack()) caps_default += (caps_default.empty() ? "" : ",") + std::string(c);

    if (interactive) std::fprintf(stderr, "Press Enter to accept the [default].\n");
    cfg.capabilities = split_list(ask_value(interactive, "Engines to enable (starter pack: llama,audiocpp,sdcpp,onnx)", caps_default));
    if (cfg.capabilities.empty()) {
        std::fprintf(stderr, "error: no engines selected\n");
        return 2;
    }

    std::string backend = cl.has("backend") ? std::string(cl.get("backend")) : (ctx.config.hardware.cpu_cores ? cfg.backend : suggested);
    backend = ask_value(interactive, "Backend (cpu, cuda, vulkan)", backend);
    if (backend != "cpu" && backend != "cuda" && backend != "vulkan") {
        std::fprintf(stderr, "error: unknown backend '%s' (expected cpu, cuda or vulkan)\n", backend.c_str());
        return 2;
    }
    cfg.backend = backend;

    // JavaScript tooling for MCP servers and skills. npm is never used: node goes with pnpm.
    std::string js = cl.has("js-runtime") ? std::string(cl.get("js-runtime")) : cfg.js_runtime;
    js = ask_value(interactive, "JavaScript tooling for MCP servers: bun, node (with pnpm) or nub", js);
    if (js != "bun" && js != "node" && js != "nub") {
        std::fprintf(stderr, "error: unknown JavaScript tooling '%s' (expected bun, node or nub)\n", js.c_str());
        return 2;
    }
    cfg.js_runtime = js;

    if (ctx.has_home) {
        // Everything lives under the home unless a flag says otherwise: no separate questions.
        cfg.workspace = cl.has("workspace") ? absolute_path(cl.get("workspace")) : std::string();
        cfg.models_dir = cl.has("models-dir") ? absolute_path(cl.get("models-dir")) : std::string();
    } else {
        const std::string ws_default = cl.has("workspace") ? std::string(cl.get("workspace")) : workspace_dir(ctx);
        cfg.workspace = absolute_path(ask_value(interactive, "Workspace directory", ws_default));
        const std::string md_default = cl.has("models-dir") ? std::string(cl.get("models-dir")) : models_dir(ctx);
        cfg.models_dir = absolute_path(ask_value(interactive, "Models directory", md_default));
    }

    // Engines: keep whatever engines.json already lists, then add the selected ones we can find.
    auto existing = load_engines(ctx.engines_path);
    std::vector<EngineEntry> engines;
    if (existing.ok) engines = std::move(existing.value);
    else std::fprintf(stderr, "warning: %s; starting a new engines.json\n", existing.error.message.c_str());

    const auto dirs = engine_search_dirs(cl.get("engines-dir"));
    std::printf("\nEngines (searching %s)\n", dirs.empty() ? "nowhere: no search directory exists" : join(dirs, ", ").c_str());
    int missing = 0;
    for (const auto &name : cfg.capabilities) {
        const std::string path = find_plugin(name, dirs);
        if (path.empty()) {
            if (const EngineEntry *e = find_engine_entry(engines, name); e && file_exists(e->path)) {
                std::printf("  %-8s kept from engines.json: %s\n", name.c_str(), e->path.c_str());
                continue;
            }
            std::printf("  %-8s not found (%s)\n", name.c_str(), plugin_file_name(name).c_str());
            ++missing;
            continue;
        }
        auto info = load_plugin_info(absolute_path(path));
        if (!info.ok) {
            std::printf("  %-8s found but failed to load: %s\n", name.c_str(), info.error.message.c_str());
            ++missing;
            continue;
        }
        EngineEntry entry{info.value.name, info.value.path, info.value.version, caps_to_names(info.value.caps), "local"};
        std::printf("  %-8s %s (%s) %s\n", entry.name.c_str(), entry.version.c_str(), join(entry.caps, ",").c_str(), entry.path.c_str());
        upsert_engine(engines, std::move(entry));
    }
    if (missing && (cl.has("download-engines") || (interactive && ask_yes_no("Download the missing engines from the GitHub release?")))) {
        for (const auto &name : cfg.capabilities) {
            if (!find_plugin(name, dirs).empty()) continue;
            if (const EngineEntry *e = find_engine_entry(engines, name); e && file_exists(e->path)) continue;
            if (auto r = pull_and_register_engine(ctx, engines, name, kDefaultReleaseRepo, "latest"); !r.ok)
                std::printf("  %-8s %s\n", name.c_str(), r.error.message.c_str());
            else --missing;
        }
    }
    if (missing)
        std::printf("\nnote: %d engine%s still missing. Run `agent engine pull NAME` to download a prebuilt one, or build it from\n"
                    "      source (e.g. -DAGENT_ENGINE_SDCPP=ON, see docs/api/engines.md) and register it with\n"
                    "      `agent engine add <path/to/agent-engine-NAME%s>`, or rerun `agent setup --engines-dir DIR`.\n",
                    missing, missing == 1 ? " is" : "s are", plugin_file_name("").substr(std::string_view("agent-engine-").size()).c_str());

    std::error_code ec;
    fs::create_directories(cfg.workspace.empty() ? workspace_dir(ctx) : cfg.workspace, ec);
    if (auto r = save_config(ctx.config_path, cfg); !r.ok) return report_error(r.error);
    if (auto r = save_engines(ctx.engines_path, engines); !r.ok) return report_error(r.error);
    ctx.config = cfg;

    std::printf("\nwrote %s\nwrote %s (%zu engine%s)\n", ctx.config_path.c_str(), ctx.engines_path.c_str(), engines.size(),
                engines.size() == 1 ? "" : "s");
    // The chosen runtimes (and uv for Python servers) are downloaded only on request: they are 20-60 MB each.
    if (ctx.has_home &&
        (cl.has("install-runtimes") || (interactive && ask_yes_no("Download the private runtimes now (" + cfg.js_runtime + ", uv)?")))) {
        std::vector<std::string> wanted = runtimes_for_js(cfg.js_runtime);
        wanted.push_back("uv");
        for (const std::string &n : wanted) {
            if (!runtime_binary_path(ctx.home, n).empty()) continue;
            std::fprintf(stderr, "downloading %s ...\n", n.c_str());
            auto r = install_runtime(ctx.home, n, "latest", agent::hub::http_options_from_env());
            if (!r.ok) std::printf("  %-5s %s\n", n.c_str(), r.error.message.c_str());
            else std::printf("  %-5s %s\n", n.c_str(), r.value.c_str());
        }
    }
    setup_models(cl, ctx, interactive);
    std::printf("\nnext: agent chat   (or `agent model search` to see more models, `agent init` inside a project)\n");
    return 0;
}

// -------------------------- agent engine --------------------------

namespace {

bool print_engine_progress(uint64_t done, uint64_t total, void *) {
    if (total) std::fprintf(stderr, "\r  %s of %s   ", format_bytes(done).c_str(), format_bytes(total).c_str());
    return true;
}

} // namespace

// Downloads the release build of `name` for this machine, checks it, and records it in `engines`.
Result<void> pull_and_register_engine(Context &ctx, std::vector<EngineEntry> &engines, const std::string &name, std::string_view repo,
                                      std::string_view tag) {
    const std::string dir = ctx.has_home ? ctx.home.engines : (fs::path(ctx.config_dir) / "engines").string();
    std::fprintf(stderr, "downloading the %s engine (%s) from github.com/%.*s ...\n", name.c_str(), std::string(tag).c_str(),
                 static_cast<int>(repo.size()), repo.data());
    auto plugin_path = pull_engine(dir, name, repo, tag, agent::hub::http_options_from_env(), print_engine_progress, nullptr);
    std::fprintf(stderr, "\n");
    if (!plugin_path.ok) return fail(plugin_path.error.code, plugin_path.error.message);
    auto info = load_plugin_info(plugin_path.value);
    if (!info.ok)
        return fail(info.error.code, "downloaded " + plugin_path.value + ", but it does not load on this machine: " + info.error.message);
    EngineEntry entry{info.value.name, plugin_path.value, info.value.version, caps_to_names(info.value.caps), "local"};
    std::printf("engine:  %s %s (%s)\npath:    %s\n", entry.name.c_str(), entry.version.c_str(), join(entry.caps, ", ").c_str(),
                entry.path.c_str());
    upsert_engine(engines, std::move(entry));
    return ok();
}

int cmd_engine(const CommandLine &cl, Context &ctx) {
    auto engines = load_engines(ctx.engines_path);
    if (!engines.ok) return report_error(engines.error);

    if (cl.subcommand == "ls") {
        if (engines.value.empty()) {
            std::printf("no engines in %s; run `agent setup` or `agent engine add <plugin>`\n", ctx.engines_path.c_str());
            return 0;
        }
        std::printf("%-8s %-12s %-8s %-8s %s\n", "NAME", "VERSION", "BACKEND", "STATUS", "PATH");
        for (const auto &e : engines.value) {
            std::printf("%-8s %-12s %-8s %-8s %s\n", e.name.c_str(), e.version.c_str(), e.backend.c_str(),
                        file_exists(e.path) ? "ok" : "MISSING", e.path.c_str());
            if (!e.caps.empty()) std::printf("         caps: %s\n", join(e.caps, ", ").c_str());
        }
        return 0;
    }

    if (cl.subcommand == "add") {
        const std::string path = absolute_path(cl.positionals[0]);
        if (!file_exists(path)) return report_error(make_error(ErrorCode::FilesystemError, "no such file: " + path));
        auto info = load_plugin_info(path);
        if (!info.ok) return report_error(info.error);
        EngineEntry entry{info.value.name, path, info.value.version, caps_to_names(info.value.caps), "local"};
        std::printf("engine:  %s\nversion: %s\ncaps:    %s\npath:    %s\n", entry.name.c_str(), entry.version.c_str(),
                    join(entry.caps, ", ").c_str(), entry.path.c_str());
        const bool replaced = find_engine_entry(engines.value, entry.name) != nullptr;
        upsert_engine(engines.value, std::move(entry));
        if (auto r = save_engines(ctx.engines_path, engines.value); !r.ok) return report_error(r.error);
        std::printf("%s %s\n", replaced ? "updated" : "added to", ctx.engines_path.c_str());
        return 0;
    }

    if (cl.subcommand == "pull") {
        const std::string name = cl.positionals[0];
        if (auto r = pull_and_register_engine(ctx, engines.value, name, cl.get("repo", kDefaultReleaseRepo), cl.get("tag", "latest"));
            !r.ok)
            return report_error(r.error);
        if (auto r = save_engines(ctx.engines_path, engines.value); !r.ok) return report_error(r.error);
        return 0;
    }

    // rm
    const std::string &name = cl.positionals[0];
    if (!remove_engine(engines.value, name))
        return report_error(make_error(ErrorCode::KeyNotFound, "no engine named '" + name + "' in " + ctx.engines_path));
    if (auto r = save_engines(ctx.engines_path, engines.value); !r.ok) return report_error(r.error);
    std::printf("removed %s (the plugin file was left in place)\n", name.c_str());
    return 0;
}

// -------------------------- agent model --------------------------

int cmd_model(const CommandLine &cl, Context &ctx) {
    const std::string dir = models_dir(ctx);
    const std::string catalog = catalog_path(ctx);

    if (cl.subcommand == "ls") {
        auto models = agent::hub::list_models(catalog);
        if (!models.ok) return report_error(models.error);
        if (models.value.empty()) {
            std::printf("no models in %s; try `agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M`\n", catalog.c_str());
            return 0;
        }
        for (const auto &m : models.value) {
            const bool is_default = m.name == ctx.config.default_model;
            std::printf("%-32s %9.1f MiB  %-14s %s:%s%s%s\n    %s\n", m.name.c_str(), static_cast<double>(m.size) / (1024.0 * 1024.0),
                        m.license.c_str(), m.repo.c_str(), m.file.c_str(), is_default ? "  (default)" : "",
                        file_exists(m.path) ? "" : "  MISSING", m.path.c_str());
            if (!m.mmproj.empty()) std::printf("    + projector %s%s\n", m.mmproj_file.c_str(), file_exists(m.mmproj) ? "" : "  MISSING");
        }
        return 0;
    }

    if (cl.subcommand == "search") {
        auto lower = [](std::string s) {
            for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        };
        const std::string needle = cl.positionals.empty() ? std::string() : lower(cl.positionals[0]);
        const std::string task = std::string(cl.get("task"));
        std::printf("%-24s %-9s %-9s %9s  %s\n", "NAME", "TASK", "ENGINE", "SIZE", "NOTES");
        size_t shown = 0;
        for (const auto &e : agent::hub::model_registry()) {
            if (!task.empty() && e.task != task) continue;
            const std::string hay =
                lower(std::string(e.alias) + " " + std::string(e.task) + " " + std::string(e.engine) + " " + std::string(e.notes));
            if (!needle.empty() && hay.find(needle) == std::string::npos) continue;
            char size[24];
            if (e.size_mib) std::snprintf(size, sizeof(size), "%u MiB", e.size_mib);
            else std::snprintf(size, sizeof(size), "-");
            std::printf("%-24s %-9.*s %-9.*s %9s  %s%.*s\n", std::string(e.alias).c_str(), static_cast<int>(e.task.size()), e.task.data(),
                        static_cast<int>(e.engine.size()), e.engine.data(), size, e.tested ? "[tested] " : "",
                        static_cast<int>(e.notes.size()), e.notes.data());
            ++shown;
        }
        if (!shown) std::printf("nothing matches; `agent model info owner/repo` lists any Hugging Face repo\n");
        else std::printf("pull one with: agent model pull <name>   ([tested] = run end to end with agent.cpp)\n");
        return 0;
    }

    if (cl.subcommand == "fit") return cmd_model_fit(cl, ctx);
    if (cl.subcommand == "recommend") return cmd_model_recommend(cl, ctx);

    if (cl.subcommand == "info") {
        // What a repo offers, so a person (or an agent) can choose a quant before pulling.
        auto info = agent::hub::fetch_repo_info(cl.positionals[0], agent::hub::http_options_from_env());
        if (!info.ok) return report_error(info.error);
        const auto &ri = info.value;
        std::printf("%s  licence: %s%s%s\n", ri.repo.c_str(), ri.license.empty() ? "unknown" : ri.license.c_str(),
                    ri.license == "other" && !ri.license_name.empty() ? (" (" + ri.license_name + ")").c_str() : "",
                    ri.gated ? "  GATED (needs HF_TOKEN)" : "");
        auto mib = [](uint64_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
        for (const auto &f : ri.files) {
            std::string lower = f.path;
            for (char &c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            const bool gguf = lower.size() > 5 && lower.compare(lower.size() - 5, 5, ".gguf") == 0;
            const bool other = lower.ends_with(".bin") || lower.ends_with(".onnx") || lower.ends_with(".safetensors");
            if (!gguf && !other) continue;
            const char *kind = lower.find("mmproj") != std::string::npos ? "projector" : (gguf ? "gguf" : "weights");
            std::printf("  %-10s %9.1f MiB  %s\n", kind, mib(f.size), f.path.c_str());
        }
        std::printf("pull one with: agent model pull %s:<QUANT or file>   (a vision / audio / OCR repo also fetches its projector)\n",
                    ri.repo.c_str());
        return 0;
    }

    if (cl.subcommand == "rm") {
        const std::string &name = cl.positionals[0];
        if (auto r = agent::hub::remove_model(name, catalog, !cl.has("keep-file")); !r.ok) return report_error(r.error);
        std::printf("removed %s%s\n", name.c_str(), cl.has("keep-file") ? " (file kept)" : "");
        if (ctx.config.default_model == name) {
            ctx.config.default_model.clear();
            if (auto r = save_config(ctx.config_path, ctx.config); !r.ok) return report_error(r.error);
        }
        return 0;
    }

    // pull
    if (cl.positionals[0].find('/') == std::string::npos) {
        // A registry name ("qwen3-4b"): pull every part (a talker and its codec, a 3D model's three files).
        const auto *entry = agent::hub::find_registry_entry(cl.positionals[0]);
        if (!entry)
            return report_error(Error{ErrorCode::KeyNotFound,
                                      "'" + cl.positionals[0] + "' is not a registry name (see `agent model search`) and not owner/repo"});
        for (size_t i = 0; i < entry->parts.size(); ++i) {
            CommandLine part = cl;
            part.positionals = {std::string(entry->parts[i].spec)};
            if (i == 0 && !cl.has("name")) part.flags.push_back(Flag{"name", std::string(entry->alias)});
            if (!entry->parts[i].include.empty()) part.flags.push_back(Flag{"include", std::string(entry->parts[i].include)});
            if (const int rc = cmd_model(part, ctx); rc != 0) return rc;
        }
        if (!entry->engine_options.empty())
            std::printf("engine options for %.*s: %.*s\n", static_cast<int>(entry->alias.size()), entry->alias.data(),
                        static_cast<int>(entry->engine_options.size()), entry->engine_options.data());
        if (!entry->notes.empty()) std::printf("note: %.*s\n", static_cast<int>(entry->notes.size()), entry->notes.data());
        return 0;
    }

    std::string spec = cl.positionals[0];
    if (cl.has("quant")) {
        // --quant Q4_K_M is the same as "owner/repo:Q4_K_M"; it can't also be given in the spec.
        if (spec.find(':') != std::string::npos)
            return report_error(Error{ErrorCode::InvalidConfig, "give the quant either as --quant or after ':' in the spec, not both"});
        spec += ":" + std::string(cl.get("quant"));
    }
    g_pull_cancel.store(false);
    std::signal(SIGINT, on_pull_signal);
    PullProgress progress;
    agent::hub::PullOptions opts;
    opts.models_dir = dir;
    opts.catalog_path = catalog;
    opts.name = std::string(cl.get("name"));
    opts.allow_restricted_license = cl.has("allow-restricted");
    for (const Flag &f : cl.flags)
        if (f.name == "include") {
            size_t from = 0; // comma separated, and the flag may repeat
            while (from <= f.value.size()) {
                const size_t comma = f.value.find(',', from);
                opts.include.push_back(f.value.substr(from, comma == std::string::npos ? std::string::npos : comma - from));
                if (comma == std::string::npos) break;
                from = comma + 1;
            }
        }
    opts.mmproj = cl.has("no-mmproj") ? "none" : std::string(cl.get("mmproj", "auto"));
    opts.on_progress = print_pull_progress;
    opts.progress_user_data = &progress;
    opts.cancel = &g_pull_cancel;

    std::fprintf(stderr, "pulling %s into %s (Ctrl-C cancels; rerun to resume)\n", spec.c_str(), dir.c_str());
    auto r = agent::hub::pull_model(spec, opts);
    if (!r.ok && r.error.code == ErrorCode::PermissionDenied && !opts.allow_restricted_license && stdin_is_tty()) {
        std::fprintf(stderr, "%s\n", r.error.message.c_str());
        if (ask_yes_no("This licence restricts use. Pull it anyway (only if you are entitled to use the model)?")) {
            opts.allow_restricted_license = true;
            r = agent::hub::pull_model(spec, opts);
        }
    }
    std::signal(SIGINT, SIG_DFL);
    std::fprintf(stderr, "\n");
    if (!r.ok) {
        if (r.error.code == ErrorCode::Cancelled) {
            std::fprintf(stderr, "cancelled; the partial download was kept, run the same command to resume\n");
            return 130;
        }
        return report_error(r.error);
    }
    std::printf("%s\n  repo:    %s\n  file:    %s\n  path:    %s\n  size:    %llu bytes\n  sha256:  %s\n  license: %s\n",
                r.value.name.c_str(), r.value.repo.c_str(), r.value.file.c_str(), r.value.path.c_str(),
                static_cast<unsigned long long>(r.value.size), r.value.sha256.empty() ? "(not published)" : r.value.sha256.c_str(),
                r.value.license.c_str());
    if (!r.value.mmproj.empty()) std::printf("  projector: %s\n", r.value.mmproj.c_str());
    if (ctx.config.default_model.empty() && engine_for_model(r.value.path) == "llama") {
        ctx.config.default_model = r.value.name;
        if (auto s = save_config(ctx.config_path, ctx.config); s.ok) std::printf("set as the default chat model\n");
    }
    return 0;
}

// -------------------------- agent run --------------------------

namespace {

bool g_run_yes = false;
StatusView *g_run_status = nullptr; // the live table, hidden while a permission prompt owns the terminal

PermissionDecision run_permission(std::string_view name, std::string_view arguments, void *) {
    PermissionDecision d = tools::default_permission_policy(name, arguments, nullptr);
    if (d == PermissionDecision::Ask && g_run_yes) return PermissionDecision::Allow;
    return d;
}

bool run_prompt(std::string_view name, std::string_view arguments, void *) {
    if (g_run_status) g_run_status->hold();
    std::fprintf(stderr, "[permission] %.*s %.*s\n", static_cast<int>(name.size()), name.data(), static_cast<int>(arguments.size()),
                 arguments.data());
    const bool answer = ask_yes_no("allow this tool call?");
    if (g_run_status) g_run_status->release();
    return answer;
}

// ModelSpec::on_load for graph models: use the run's store (one SQLite connection for the whole run)
// and the MCP tools load_graph_json connected on the run's Session.
Result<void> share_home(Session &session, const ModelSpec &, void *user_data) {
    const Session &home = *static_cast<const Session *>(user_data);
    session.config.memory = home.config.memory;
    session.memory_state = home.memory_state;
    bool added = false;
    for (const Tool &t : home.config.tools) {
        if (t.name.find('.') == std::string_view::npos || session.tool_by_name.count(std::string(t.name))) continue; // "<server>.<tool>"
        session.config.tools.push_back(t);
        added = true;
    }
    if (added) rebuild_tool_index(session);
    return ok();
}

// Registers a model named in the graph (catalogue name or file path, as written) with the registry.
Result<void> register_graph_model(const Context &ctx, ModelRegistry &registry, const std::string &name, const Config &base, bool inprocess,
                                  Session *home) {
    if (has_model(registry, name)) return ok();
    auto model = resolve_model(name, {}, catalog_refs(ctx), path_exists);
    if (!model.ok) return fail(model.error.code, "model '" + name + "': " + model.error.message);
    auto engine = load_engine_for(ctx, {}, model.value.path);
    if (!engine.ok) return fail(engine.error.code, "model '" + name + "': " + engine.error.message);

    ModelSpec spec;
    spec.config = base;
    spec.config.provider = AiProvider::Engine;
    spec.config.model = model.value.path;
    if (auto cat = agent::hub::load_catalog(catalog_path(ctx)); cat.ok)
        for (const auto &entry : cat.value)
            if (entry.name == model.value.name && entry.path == model.value.path) spec = agent::hub::model_spec_from_entry(entry, base);
    spec.name = name;
    spec.config.engine = engine.value.name;
    spec.config.engine_isolation = cli_engine_isolation(inprocess, engine.value.isolation);
    spec.config.engine_sandbox.gpu = cli_engine_gpu(engine.value.backend);
    apply_sandbox_choice(spec.config);
    spec.on_load = share_home;
    spec.on_load_user_data = home;
    return register_model(registry, std::move(spec));
}

const char *status_name(GraphRunStatus s) {
    switch (s) {
    case GraphRunStatus::Pending: return "pending";
    case GraphRunStatus::Running: return "running";
    case GraphRunStatus::Completed: return "completed";
    case GraphRunStatus::Interrupted: return "interrupted";
    case GraphRunStatus::Failed: return "failed";
    }
    return "?";
}

} // namespace

// agent run graph.json [--set key=value ...] [--thread ID] [--resume ID [--input TEXT]]
int cmd_run(const CommandLine &cl, Context &ctx) {
    const std::string &graph_path = cl.positionals[0];
    g_run_yes = cl.has("yes");

    // Engine/model hints come from the file; flags override them.
    auto hints = parse_graph_json(graph_path);
    if (!hints.ok) return report_error(hints.error);
    for (const auto &w : hints.value.warnings) std::fprintf(stderr, "[warning] %s\n", w.c_str());

    // Agent nodes with a "model" run on that model from the registry (memory_budget_mb, LRU unload). The
    // default model is loaded only when some Agent node has no "model" (or there is no Agent node);
    // otherwise the run's own Session is a model-less store/tool host.
    std::vector<std::string> node_models;
    bool needs_default = true;
    bool any_agent = false;
    for (const GraphNode &n : hints.value.graph.nodes) {
        if (n.kind != GraphNodeKind::Agent) continue;
        if (!any_agent) needs_default = false;
        any_agent = true;
        if (n.model.empty()) needs_default = true;
        else if (std::find(node_models.begin(), node_models.end(), n.model) == node_models.end()) node_models.push_back(n.model);
    }

    // With a memory budget (or --swap-models) and nodes that name their own models, the default model is not pinned for
    // the whole run: its nodes use it through the registry like any other model, so it is unloaded while a node runs
    // a different one (a large main model next to small sub-agent models that do not fit in memory together).
    std::string swap_default_path;
    if (needs_default && !node_models.empty() && (ctx.config.memory_budget_mb > 0 || cl.has("swap-models"))) {
        auto model = resolve_model(cl.get("model", hints.value.model), ctx.config.default_model, catalog_refs(ctx), path_exists);
        if (!model.ok) return report_error(model.error);
        swap_default_path = model.value.path;
        node_models.push_back(swap_default_path);
        needs_default = false;
    }

    Config base;
    base.workspace_dir = workspace_dir(ctx, cl.get("workspace"));
    base.context_window = ctx.config.context_window;
    base.max_tokens = ctx.config.max_tokens;
    base.temperature = ctx.config.temperature;
    install_cli_hooks(base, ctx);
    base.permission_check = run_permission;
    base.permission_prompt = run_prompt;

    ModelRegistry registry(ModelBudget{.ram_bytes = static_cast<uint64_t>(std::max<int64_t>(0, ctx.config.memory_budget_mb)) << 20});
    ModelLease default_lease; // held for the whole run: the run's Session
    Session plain;            // the run's Session when no default model is needed
    Session *session = &plain;
    std::string model_label = "no default model";
    std::string engine_label = "-";
    if (needs_default) {
        std::string_view model_spec = cl.get("model", hints.value.model);
        auto model = resolve_model(model_spec, ctx.config.default_model, catalog_refs(ctx), path_exists);
        if (!model.ok) return report_error(model.error);
        auto engine = load_engine_for(ctx, cl.get("engine", hints.value.engine), model.value.path);
        if (!engine.ok) return report_error(engine.error);
        ModelSpec spec;
        spec.name = model.value.name;
        spec.config = base;
        spec.config.provider = AiProvider::Engine;
        spec.config.engine = engine.value.name;
        spec.config.engine_isolation = cli_engine_isolation(cl.has("inprocess"), engine.value.isolation);
        spec.config.engine_sandbox.gpu = cli_engine_gpu(engine.value.backend);
        apply_sandbox_choice(spec.config);
        spec.config.engine_options = catalog_engine_options(ctx, model.value.path, std::string(cl.get("engine-options")));
        spec.config.model = model.value.path;
        spec.config.memory = {.store = store::sqlite_store()};
        if (auto r = register_model(registry, std::move(spec)); !r.ok) return report_error(r.error);
        auto lease = acquire_model(registry, model.value.name);
        if (!lease.ok) return report_error(lease.error);
        default_lease = std::move(lease.value);
        session = default_lease.session;
        model_label = model.value.name;
        engine_label = engine.value.name;
    } else {
        Config cfg = base;
        cfg.provider = AiProvider::Mock;
        cfg.memory = {.store = store::sqlite_store()};
        auto s = init(cfg);
        if (!s.ok) return report_error(s.error);
        plain = std::move(s.value);
    }
    for (const auto &name : node_models)
        if (auto r = register_graph_model(ctx, registry, name, base, cl.has("inprocess"), session); !r.ok) return report_error(r.error);

    auto graph = load_graph_json(*session, graph_path);
    if (!graph.ok) return report_error(graph.error);
    if (!swap_default_path.empty()) {
        for (GraphNode &n : graph.value.graph.nodes)
            if (n.kind == GraphNodeKind::Agent && n.model.empty()) n.model = swap_default_path;
        model_label = "swapping (" + std::filesystem::path(swap_default_path).filename().string() + " is the default)";
    }

    // Checkpoints and the event log go to <workspace>/memory/checkpoints.sqlite3, so --resume works
    // across invocations; "checkpointer": "memory" in the file keeps them in this process only.
    const bool memory_checkpoints = graph.value.checkpointer == "memory";
    CheckpointerConfig cp_config;
    if (!memory_checkpoints) cp_config.path = (fs::path(base.workspace_dir) / "memory" / "checkpoints.sqlite3").string();
    auto checkpointer = open_checkpointer(memory_checkpoints ? in_memory_checkpointer() : store::sqlite_checkpointer(), cp_config);
    if (!checkpointer.ok) return report_error(checkpointer.error);

    GraphRun run;
    run.models = &registry;
    run.checkpointer = checkpointer.value;
    const std::string_view resume_id = cl.get("resume");
    // A new run gets its own thread: continuing a thread reuses its state and histories, so a fixed default would mix unrelated graphs.
    // Pass --thread to continue one on purpose.
    const std::string fresh_thread =
        (graph.value.graph.name.empty() ? std::string("run") : graph.value.graph.name) + "-" +
        std::to_string(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    run.thread_id = std::string(resume_id.empty() ? cl.get("thread", fresh_thread) : resume_id);
    for (const Flag &f : cl.flags) {
        if (f.name != "set") continue;
        const size_t eq = f.value.find('=');
        if (eq == std::string::npos || eq == 0) {
            std::fprintf(stderr, "agent run: --set expects key=value, got '%s'\n", f.value.c_str());
            return 2;
        }
        set_state(run.state, f.value.substr(0, eq), f.value.substr(eq + 1));
    }

    std::fprintf(stderr, "running %s (thread %s) with %s on %s\n", graph_path.c_str(), run.thread_id.c_str(), model_label.c_str(),
                 engine_label.c_str());
    for (const auto &name : node_models) std::fprintf(stderr, "  node model: %s\n", name.c_str());

    // Node status: a table redrawn in place on a terminal, one line per event otherwise (see status_view.hpp).
    // It is drawn on stderr, like the progress lines above; stdout keeps the final status and state.
    const StatusMode status_mode =
        choose_status_mode(cl.has("json"), stream_is_tty(stdout), stream_is_tty(stderr), std::getenv("TERM"), std::getenv("NO_COLOR"));
    StatusMode mode = status_mode;
    if (mode == StatusMode::Live && !enable_ansi(stderr)) mode = StatusMode::Plain;
    StatusView status(stderr, mode, graph.value.graph, mode == StatusMode::Live ? terminal_width(stderr) : 0);
    run.on_event = StatusView::on_graph_event;
    run.on_event_user_data = &status;
    g_run_status = &status;

    auto res = resume_id.empty() ? run_graph(*session, graph.value.graph, run)
                                 : resume_graph(*session, graph.value.graph, resume_id, run, cl.get("input"));
    g_run_status = nullptr;
    status.finish();
    std::printf("status: %s\n", status_name(run.status));
    if (run.status == GraphRunStatus::Interrupted) {
        std::printf("interrupted at %s: %s\n", run.interrupt_node.c_str(), run.interrupt_reason.c_str());
        for (const GraphInterrupt &i : run.interrupts)
            if (!i.branch.empty()) std::printf("  branch %s paused at %s: %s\n", i.branch.c_str(), i.node.c_str(), i.reason.c_str());
        for (const ToolCall &tc : run.pending_tool_calls)
            std::printf("  pending tool call: %s %s\n", tc.name.c_str(), tc.arguments.c_str());
        std::printf("resume with: agent run %s --resume %s --input \"<reply or approve>\"\n", graph_path.c_str(), run.thread_id.c_str());
    }
    for (const auto &[key, value] : run.state.values) std::printf("%s = %s\n", key.c_str(), value.c_str());
    if (!res.ok) return report_error(res.error);
    return run.status == GraphRunStatus::Completed || run.status == GraphRunStatus::Interrupted ? 0 : 1;
}

} // namespace agent::cli
