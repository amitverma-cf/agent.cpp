// agent model fit / agent model recommend: will a model run on this machine, and about how fast. See fit.hpp.

#include "commands.hpp"
#include "doctor.hpp"
#include "fit.hpp"

#include <agent-cpp/hub.hpp>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <simdjson.h>

namespace agent::cli {

namespace {

namespace fs = std::filesystem;

constexpr uint64_t kHeaderBytes = 4u << 20;     // the first try
constexpr uint64_t kHeaderBytesMax = 24u << 20; // a model with a very large tokenizer

std::string lower_copy(std::string s) {
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string read_prefix(const std::string &path, uint64_t bytes) {
    std::ifstream f(path, std::ios::binary);
    std::string out(static_cast<size_t>(bytes), '\0');
    f.read(out.data(), static_cast<std::streamsize>(bytes));
    out.resize(static_cast<size_t>(f.gcount()));
    return out;
}

std::string resolve_url(const agent::hub::HttpOptions &http, const std::string &repo, const std::string &file) {
    // Path segments of a repo file are percent-encoded one by one; plain names pass through unchanged.
    auto encode = [](const std::string &s) {
        std::string out;
        for (unsigned char c : s) {
            if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '/' || c == '~') out += static_cast<char>(c);
            else {
                char buf[4];
                std::snprintf(buf, sizeof(buf), "%%%02X", c);
                out += buf;
            }
        }
        return out;
    };
    return http.endpoint + "/" + repo + "/resolve/main/" + encode(file);
}

// The header of the model at `url`, growing the range once if the hyper-parameters were not in the first bytes.
agent::Result<GgufInfo> remote_header(const std::string &url, const agent::hub::HttpOptions &http) {
    for (uint64_t bytes : {kHeaderBytes, kHeaderBytesMax}) {
        auto body = agent::hub::http_get_range(url, http, 0, bytes);
        if (!body.ok) return agent::fail<GgufInfo>(body.error.code, body.error.message);
        GgufInfo info = parse_gguf_header(body.value);
        if (!info.valid) return agent::fail<GgufInfo>(agent::ErrorCode::DecodeFailed, "not a GGUF file: " + url);
        if (info.has_shape() || body.value.size() < bytes) return agent::ok(std::move(info));
    }
    return agent::fail<GgufInfo>(agent::ErrorCode::DecodeFailed, "the GGUF header is larger than " + format_bytes(kHeaderBytesMax));
}

KvType kv_from_flag(std::string_view v) { return v == "f16" ? KvType::F16 : v == "q4" ? KvType::Q4 : KvType::Q8; }

void print_machine(const Machine &m) {
    std::printf("this machine: %d cores, RAM %s (%s free)", m.cores, format_bytes(m.ram_total).c_str(),
                m.ram_free ? format_bytes(m.ram_free).c_str() : "unknown");
    if (m.vram_total) std::printf(", GPU memory %s", format_bytes(m.vram_total).c_str());
    if (m.mem_bandwidth > 0) std::printf(", memory bandwidth about %.0f GB/s", m.mem_bandwidth / 1e9);
    std::printf("\n");
}

std::string speed_text(const FitResult &r) {
    char buf[48];
    if (!r.fits) return "does not fit";
    if (r.tokens_per_second > 0) {
        std::snprintf(buf, sizeof(buf), "about %.0f tok/s", r.tokens_per_second);
        return buf;
    }
    return r.on_gpu ? "fits on the GPU" : "fits";
}

struct Candidate {
    std::string repo;
    uint64_t downloads = 0;
};

std::vector<Candidate> parse_search(std::string_view json) {
    std::vector<Candidate> out;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json.data(), json.size()).get(doc)) return out;
    simdjson::dom::array arr;
    if (doc.get_array().get(arr)) return out;
    for (simdjson::dom::element e : arr) {
        std::string_view id;
        int64_t downloads = 0;
        if (e["id"].get_string().get(id)) continue;
        if (e["downloads"].get_int64().get(downloads)) downloads = 0;
        out.push_back({std::string(id), static_cast<uint64_t>(std::max<int64_t>(0, downloads))});
    }
    return out;
}

// The file to try for a repo: the chosen quant, else the smallest non-projector GGUF.
const agent::hub::RemoteFile *pick_gguf(const agent::hub::RepoInfo &info, const std::string &quant) {
    const agent::hub::RemoteFile *best = nullptr;
    for (const auto &f : info.files) {
        const std::string name = lower_copy(f.path);
        if (!name.ends_with(".gguf") || name.find("mmproj") != std::string::npos || name.find("-0000") != std::string::npos) continue;
        if (!quant.empty() && name.find(lower_copy(quant)) != std::string::npos) return &f;
        if (!best || (f.size && f.size < best->size)) best = &f;
    }
    return quant.empty() ? best : nullptr;
}

} // namespace

int cmd_model_fit(const CommandLine &cl, Context &ctx) {
    const std::string spec = cl.positionals[0];
    const int context = cl.has("ctx") ? std::max(256, std::atoi(std::string(cl.get("ctx")).c_str())) : ctx.config.context_window;
    const KvType kv = kv_from_flag(cl.get("kv"));
    const auto http = agent::hub::http_options_from_env();

    GgufInfo info;
    uint64_t file_bytes = 0;
    std::string what = spec;
    std::error_code ec;
    if (fs::is_regular_file(spec, ec)) {
        info = parse_gguf_header(read_prefix(spec, kHeaderBytesMax));
        file_bytes = fs::file_size(spec, ec);
        if (!info.valid) return report_error(make_error(ErrorCode::DecodeFailed, spec + " is not a GGUF file"));
    } else {
        auto ms = agent::hub::parse_spec(spec);
        if (!ms.ok) return report_error(ms.error);
        auto repo = agent::hub::fetch_repo_info(ms.value.repo, http);
        if (!repo.ok) return report_error(repo.error);
        auto file = agent::hub::select_file(repo.value, ms.value);
        if (!file.ok) return report_error(file.error);
        file_bytes = file.value.size;
        if (auto set = agent::hub::split_set(repo.value, file.value); set.ok && set.value.size() > 1) {
            file_bytes = 0;
            for (const auto &shard : set.value) file_bytes += shard.size;
        }
        what = ms.value.repo + ":" + file.value.path;
        auto header = remote_header(resolve_url(http, ms.value.repo, file.value.path), http);
        if (!header.ok) return report_error(header.error);
        info = header.value;
    }

    const Machine machine = probe_machine(true);
    const FitResult r = estimate_fit(info, file_bytes, machine, context, kv);
    std::printf("%s  (%s, %llu layers, %s)\n", what.c_str(), info.architecture.empty() ? "unknown architecture" : info.architecture.c_str(),
                static_cast<unsigned long long>(info.block_count), format_bytes(file_bytes).c_str());
    print_machine(machine);
    std::printf("memory needed: %s  = weights %s + KV cache %s (%d tokens of context) + buffers %s\n", format_bytes(r.need).c_str(),
                format_bytes(r.weights).c_str(), format_bytes(r.kv_cache).c_str(), context, format_bytes(r.buffers).c_str());
    std::printf("verdict: %s%s%s\n", speed_text(r).c_str(), r.note.empty() ? "" : "  (", r.note.empty() ? "" : (r.note + ")").c_str());
    return r.fits ? 0 : 1;
}

int cmd_model_recommend(const CommandLine &cl, Context &ctx) {
    const auto http = agent::hub::http_options_from_env();
    const std::string query = cl.positionals.empty() ? std::string("instruct") : cl.positionals[0];
    const std::string quant = cl.has("quant") ? std::string(cl.get("quant")) : std::string("Q4_K_M");
    const int context = cl.has("ctx") ? std::max(256, std::atoi(std::string(cl.get("ctx")).c_str())) : ctx.config.context_window;
    const int limit = cl.has("limit") ? std::max(1, std::atoi(std::string(cl.get("limit")).c_str())) : 8;

    const Machine machine = probe_machine(true);
    print_machine(machine);
    std::printf("searching Hugging Face for \"%s\" (GGUF, %s, most downloaded first) ...\n", query.c_str(), quant.c_str());
    auto found = agent::hub::http_get_text(
        http.endpoint + "/api/models?search=" + query + "&filter=gguf&sort=downloads&direction=-1&limit=40", http);
    if (!found.ok) return report_error(found.error);
    const auto candidates = parse_search(found.value);
    if (candidates.empty()) {
        std::printf("nothing found\n");
        return 1;
    }

    struct Row {
        std::string repo, file;
        uint64_t downloads = 0;
        FitResult fit;
    };
    std::vector<Row> rows;
    int tried = 0;
    for (const Candidate &c : candidates) {
        if (static_cast<int>(rows.size()) >= limit || tried >= 24) break;
        ++tried;
        auto repo = agent::hub::fetch_repo_info(c.repo, http);
        if (!repo.ok || repo.value.gated) continue;
        const auto *file = pick_gguf(repo.value, quant);
        if (!file || !file->size) continue;
        // A model that cannot fit is rejected from its size alone, before any header is fetched.
        if (file->size > std::max(machine.ram_total, machine.vram_total)) continue;
        auto header = remote_header(resolve_url(http, c.repo, file->path), http);
        if (!header.ok) continue;
        Row row{c.repo, file->path, c.downloads, estimate_fit(header.value, file->size, machine, context)};
        if (row.fit.fits) rows.push_back(std::move(row));
    }
    if (rows.empty()) {
        std::printf("no model of this kind fits this machine at %s; try a smaller quant (--quant Q3_K_M) or a smaller context (--ctx)\n",
                    quant.c_str());
        return 1;
    }
    // Fastest first among what fits; the speed is the point of a recommendation.
    std::stable_sort(rows.begin(), rows.end(),
                     [](const Row &a, const Row &b) { return a.fit.tokens_per_second > b.fit.tokens_per_second; });
    std::printf("\n%-52s %10s %10s  %s\n", "MODEL", "SIZE", "NEEDS", "SPEED");
    for (const Row &r : rows)
        std::printf("%-52s %10s %10s  %s\n", (r.repo + ":" + quant).c_str(), format_bytes(r.fit.weights).c_str(),
                    format_bytes(r.fit.need).c_str(), speed_text(r.fit).c_str());
    std::printf("\npull one with: agent model pull <model>   (speeds are estimates from memory bandwidth; `agent model fit <model>` shows "
                "the arithmetic)\n");
    return 0;
}

} // namespace agent::cli
