#include "speed.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>

namespace agent::cli {

namespace {

std::string lowered(std::string_view s) {
    std::string out(s);
    for (char &c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// "prompt lookup", "Prompt-Lookup" and "prompt_lookup" are the same thing.
std::string spec_name(std::string_view value) {
    std::string v = lowered(value);
    for (char &c : v)
        if (c == ' ' || c == '-') c = '_';
    if (v == "draft_model") v = "draft";
    return v;
}

bool whole_number(std::string_view value, long lowest, long highest, long &out) {
    const std::string text(value);
    char *end = nullptr;
    const long v = std::strtol(text.c_str(), &end, 10);
    if (text.empty() || *end != 0 || v < lowest || v > highest) return false;
    out = v;
    return true;
}

} // namespace

std::string check_setting_value(std::string_view key, std::string_view value) {
    long n = 0;
    if (key == "temperature") {
        const std::string text(value);
        char *end = nullptr;
        const float v = std::strtof(text.c_str(), &end);
        if (text.empty() || end == text.c_str() || *end != 0 || v < 0.0f || v > 2.0f) return "a number from 0 to 2";
        return {};
    }
    if (key == "max_tokens") return whole_number(value, 16, 1048576, n) ? std::string() : "a whole number of at least 16";
    if (key == "context_window") return whole_number(value, 2048, 1048576, n) ? std::string() : "a whole number of at least 2048";
    if (key == "mode") return value == "ask" || value == "plan" || value == "auto" ? std::string() : "choose ask, plan or auto";
    if (key == "kv_cache")
        return value == "auto" || value == "f16" || value == "q8_0" || value == "q4_0" ? std::string() : "choose auto, f16, q8_0 or q4_0";
    if (key == "speculative") {
        const std::string v = spec_name(value);
        return v == "off" || v == "prompt_lookup" || v == "draft" ? std::string() : "choose off, prompt lookup or draft model";
    }
    if (key == "mtp") return value == "on" || value == "off" ? std::string() : "choose on or off";
    if (key == "draft_model")
        return value.find_first_of("\"\\n") == std::string_view::npos ? std::string() : "a model name from agent model ls";
    return {};
}

bool set_speed_setting(CliConfig &c, std::string_view key, std::string_view value) {
    if (key == "kv_cache") c.kv_cache = std::string(value);
    else if (key == "speculative") c.speculative = spec_name(value);
    else if (key == "draft_model") c.draft_model = value == "(none)" ? std::string() : std::string(value);
    else if (key == "mtp") c.mtp = value == "on";
    else return false;
    return true;
}

std::string speed_setting_text(const CliConfig &c, std::string_view key) {
    if (key == "kv_cache") return c.kv_cache;
    if (key == "speculative") return c.speculative == "prompt_lookup" ? "prompt lookup" : c.speculative == "draft" ? "draft model" : "off";
    if (key == "draft_model") return c.draft_model.empty() ? "(none)" : c.draft_model;
    if (key == "mtp") return c.mtp ? "on" : "off";
    return {};
}

std::string speed_engine_options(const CliConfig &c, std::string_view engine, std::string_view user_options, std::string_view draft_path) {
    std::string options(user_options);
    if (engine != "llama") return options;
    if (c.kv_cache != "auto" && !c.kv_cache.empty()) options = with_option(options, "kv_cache", "\"" + json_escape(c.kv_cache) + "\"");
    if (c.speculative == "off") options = with_option(options, "speculative", "\"off\"");
    if (c.speculative == "draft" && !draft_path.empty()) {
        options = with_option(options, "speculative", "\"draft\"");
        options = with_option(options, "draft_model", "\"" + json_escape(draft_path) + "\"");
    }
    if (!c.mtp) options = with_option(options, "mtp", "false");
    return options;
}

std::vector<std::string> draft_model_candidates(std::span<const std::pair<std::string, uint64_t>> installed, std::string_view in_use) {
    std::vector<std::string> out;
    for (const auto &[name, size] : installed) {
        const std::string low = lowered(name);
        bool other = name == in_use || size == 0 || size > (2ull << 30); // a draft model is small: under 2 GiB
        for (const char *w : {"embed", "bge-", "minilm", "rerank", "e5-", "gte-", "whisper", "tts", "mmproj", "vae", "parakeet"})
            if (low.find(w) != std::string::npos) other = true;
        if (!other) out.push_back(name);
    }
    return out;
}

std::string speculation_note(const TokenUsage &u) {
    char buf[96];
    const char *kind = u.spec_kind == 3 ? "MTP" : u.spec_kind == 2 ? "draft model" : u.spec_kind == 1 ? "lookup" : nullptr;
    if (!kind) return {};
    if (u.spec_drafted <= 0) {
        std::snprintf(buf, sizeof(buf), "%s on", kind);
        return buf;
    }
    std::snprintf(buf, sizeof(buf), "%s on, %d of %d drafted tokens kept", kind, u.spec_accepted, u.spec_drafted);
    return buf;
}

std::string context_text(int used, int total) {
    if (total <= 0) return {};
    return std::to_string(used) + "/" + std::to_string(total) + " tokens";
}

} // namespace agent::cli
