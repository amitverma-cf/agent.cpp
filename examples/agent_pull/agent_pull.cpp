// agent_pull: download models from Hugging Face into the local catalogue (models.json).
//
//   agent_pull <owner/repo[:file-or-QUANT]> [--dir DIR] [--name NAME] [--allow-restricted]
//   agent_pull --list [--dir DIR]
//   agent_pull --rm NAME [--dir DIR] [--keep-file]

#include <agent-cpp/hub.hpp>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>

namespace {

std::atomic<bool> g_cancel{false};

void on_signal(int) { g_cancel.store(true); }

bool print_progress(uint64_t done, uint64_t total, void *) {
    static uint64_t last_reported = 0;
    if (done - last_reported < (1u << 20) && done != total) return true;
    last_reported = done;
    const double mib = static_cast<double>(done) / (1024.0 * 1024.0);
    if (total) {
        const double pct = 100.0 * static_cast<double>(done) / static_cast<double>(total);
        std::fprintf(stderr, "\r  %.1f / %.1f MiB (%5.1f%%)", mib, static_cast<double>(total) / (1024.0 * 1024.0), pct);
    } else {
        std::fprintf(stderr, "\r  %.1f MiB", mib);
    }
    std::fflush(stderr);
    return true;
}

int usage() {
    std::fprintf(stderr, "usage:\n"
                         "  agent_pull <owner/repo[:file-or-QUANT]> [--dir DIR] [--name NAME] [--allow-restricted]\n"
                         "  agent_pull --list [--dir DIR]\n"
                         "  agent_pull --rm NAME [--dir DIR] [--keep-file]\n"
                         "env: HF_TOKEN, HF_ENDPOINT, HTTPS_PROXY, NO_PROXY, AGENT_CA_BUNDLE/SSL_CERT_FILE, AGENT_MODELS_DIR\n");
    return 2;
}

int report(const agent::Error &e) {
    std::fprintf(stderr, "\nerror: %s\n", e.message.c_str());
    return 1;
}

} // namespace

int main(int argc, char **argv) {
    std::string spec, dir, name, rm_name;
    bool list = false, allow_restricted = false, keep_file = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        auto next = [&](std::string &out) {
            if (i + 1 >= argc) return false;
            out = argv[++i];
            return true;
        };
        if (a == "--dir") { if (!next(dir)) return usage(); }
        else if (a == "--name") { if (!next(name)) return usage(); }
        else if (a == "--rm") { if (!next(rm_name)) return usage(); }
        else if (a == "--list" || a == "-l") list = true;
        else if (a == "--allow-restricted") allow_restricted = true;
        else if (a == "--keep-file") keep_file = true;
        else if (a == "-h" || a == "--help") return usage();
        else if (!a.empty() && a.front() == '-') return usage();
        else if (spec.empty()) spec = std::string(a);
        else return usage();
    }

    if (dir.empty()) dir = agent::hub::default_models_dir();
    const std::string catalog = (std::filesystem::path(dir) / "models.json").string();

    if (list) {
        auto models = agent::hub::list_models(catalog);
        if (!models.ok) return report(models.error);
        if (models.value.empty()) std::printf("no models in %s\n", catalog.c_str());
        for (const auto &m : models.value)
            std::printf("%-32s %8.1f MiB  %-14s %s:%s\n    %s\n", m.name.c_str(), static_cast<double>(m.size) / (1024.0 * 1024.0), m.license.c_str(),
                        m.repo.c_str(), m.file.c_str(), m.path.c_str());
        return 0;
    }
    if (!rm_name.empty()) {
        auto r = agent::hub::remove_model(rm_name, catalog, !keep_file);
        if (!r.ok) return report(r.error);
        std::printf("removed %s\n", rm_name.c_str());
        return 0;
    }
    if (spec.empty()) return usage();

    std::signal(SIGINT, on_signal);
    agent::hub::PullOptions opts;
    opts.models_dir = dir;
    opts.catalog_path = catalog;
    opts.name = name;
    opts.allow_restricted_license = allow_restricted;
    opts.on_progress = print_progress;
    opts.cancel = &g_cancel;

    std::fprintf(stderr, "pulling %s into %s\n", spec.c_str(), dir.c_str());
    auto r = agent::hub::pull_model(spec, opts);
    if (!r.ok) return report(r.error);
    std::fprintf(stderr, "\n");
    std::printf("%s\n  repo:    %s\n  file:    %s\n  path:    %s\n  size:    %llu bytes\n  sha256:  %s\n  license: %s\n", r.value.name.c_str(),
                r.value.repo.c_str(), r.value.file.c_str(), r.value.path.c_str(), static_cast<unsigned long long>(r.value.size),
                r.value.sha256.empty() ? "(not published)" : r.value.sha256.c_str(), r.value.license.c_str());
    return 0;
}
