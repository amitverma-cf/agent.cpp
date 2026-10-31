// agent setup / engine / model / run, plus helpers shared with chat.cpp.

#include "commands.hpp"

#include <agent-cpp/engine_abi.h>
#include <agent-cpp/hub.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
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
    return ctx.config.models_dir.empty() ? agent::hub::default_models_dir() : ctx.config.models_dir;
}

std::string catalog_path(const Context &ctx) { return (fs::path(models_dir(ctx)) / "models.json").string(); }

std::string workspace_dir(const Context &ctx, std::string_view override_dir) {
    if (!override_dir.empty()) return std::string(override_dir);
    return ctx.config.workspace.empty() ? default_workspace_dir() : ctx.config.workspace;
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
    return loaded;
}

// -------------------------- agent setup --------------------------

int cmd_setup(const CommandLine &cl, Context &ctx) {
    const bool interactive = stdin_is_tty() && !cl.has("yes");
    const HardwareInfo hw = detect_hardware();

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
    cfg.capabilities = split_list(ask_value(interactive, "Engines to enable (starter pack: llama,sherpa,sdcpp,onnx)", caps_default));
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

    const std::string ws_default = cl.has("workspace") ? std::string(cl.get("workspace")) : workspace_dir(ctx);
    cfg.workspace = absolute_path(ask_value(interactive, "Workspace directory", ws_default));
    const std::string md_default = cl.has("models-dir") ? std::string(cl.get("models-dir")) : models_dir(ctx);
    cfg.models_dir = absolute_path(ask_value(interactive, "Models directory", md_default));

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
    if (missing)
        std::printf("\nnote: downloading prebuilt engine plugins from GitHub releases is not available yet. Build the missing\n"
                    "      engines from source (e.g. -DAGENT_ENGINE_SDCPP=ON, see docs/api/engines.md) and register them with\n"
                    "      `agent engine add <path/to/agent-engine-NAME%s>`, or rerun `agent setup --engines-dir DIR`.\n",
                    plugin_file_name("").substr(std::string_view("agent-engine-").size()).c_str());

    std::error_code ec;
    fs::create_directories(cfg.workspace, ec);
    if (auto r = save_config(ctx.config_path, cfg); !r.ok) return report_error(r.error);
    if (auto r = save_engines(ctx.engines_path, engines); !r.ok) return report_error(r.error);
    ctx.config = cfg;

    std::printf("\nwrote %s\nwrote %s (%zu engine%s)\n", ctx.config_path.c_str(), ctx.engines_path.c_str(), engines.size(),
                engines.size() == 1 ? "" : "s");
    std::printf("\nnext: agent model pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M && agent chat\n");
    return 0;
}

// -------------------------- agent engine --------------------------

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
        }
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
    const std::string &spec = cl.positionals[0];
    g_pull_cancel.store(false);
    std::signal(SIGINT, on_pull_signal);
    PullProgress progress;
    agent::hub::PullOptions opts;
    opts.models_dir = dir;
    opts.catalog_path = catalog;
    opts.name = std::string(cl.get("name"));
    opts.allow_restricted_license = cl.has("allow-restricted");
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
    std::printf("%s\n  repo:    %s\n  file:    %s\n  path:    %s\n  size:    %llu bytes\n  sha256:  %s\n  license: %s\n", r.value.name.c_str(),
                r.value.repo.c_str(), r.value.file.c_str(), r.value.path.c_str(), static_cast<unsigned long long>(r.value.size),
                r.value.sha256.empty() ? "(not published)" : r.value.sha256.c_str(), r.value.license.c_str());
    if (ctx.config.default_model.empty() && engine_for_model(r.value.path) == "llama") {
        ctx.config.default_model = r.value.name;
        if (auto s = save_config(ctx.config_path, ctx.config); s.ok) std::printf("set as the default chat model\n");
    }
    return 0;
}

// -------------------------- agent run --------------------------

int cmd_run(const CommandLine &cl, Context &) {
    std::fprintf(stderr, "agent run %s: not implemented yet: depends on the agent.json loader\n", cl.positionals[0].c_str());
    return 1;
}

} // namespace agent::cli
