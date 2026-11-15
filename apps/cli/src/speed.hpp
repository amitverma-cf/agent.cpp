#pragma once

// Speed: the settings that make answers faster or slower (KV cache type, speculative decoding, MTP, a draft model), the checks every
// setting goes through, the engine options they turn into, and the short text the status line shows about them.

#include "cli_core.hpp"

#include <agent-cpp/agent.hpp>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace agent::cli {

// "" when `value` is fine for the setting `key`, else what is expected ("a number from 0 to 2"). Covers temperature, max_tokens,
// context_window, mode, kv_cache, speculative, mtp and draft_model; a key it does not know passes. The Settings screen and the
// settings tool both call it, so they accept and refuse the same things.
std::string check_setting_value(std::string_view key, std::string_view value);

// Stores kv_cache, speculative, draft_model or mtp in the config (not saved to disk). False when `key` is not one of them.
// `value` has been through check_setting_value; "prompt lookup" and "prompt_lookup" both work.
bool set_speed_setting(CliConfig &config, std::string_view key, std::string_view value);

// The text the Settings screen shows for a speed setting ("prompt lookup", "on", "q8_0").
std::string speed_setting_text(const CliConfig &config, std::string_view key);

// The engine options of a llama model: `user_options` (the --engine-options JSON) plus what the speed settings ask for. Options the
// user wrote win. `draft_path` is the file of config.draft_model ("" when it is not installed). Other engines get `user_options`.
std::string speed_engine_options(const CliConfig &config, std::string_view engine, std::string_view user_options,
                                 std::string_view draft_path);

// Installed models that can serve as a draft model: small chat models (not the one in use, no embedding or speech models).
std::vector<std::string> draft_model_candidates(std::span<const std::pair<std::string, uint64_t>> installed, std::string_view in_use);

// "MTP on 87% of 40 drafted" / "lookup 12 of 30 drafted" / "": what speculative decoding did in the reply behind `usage`.
std::string speculation_note(const TokenUsage &usage);
// "951/8192 tokens": what the prompt takes of the model's window.
std::string context_text(int used, int total);

} // namespace agent::cli
