#include "cli_core.hpp"

#include <catch_amalgamated.hpp>
#include <filesystem>
#include <string>
#include <vector>

using namespace agent::cli;

namespace {

agent::Result<CommandLine> parse(std::initializer_list<std::string_view> args) {
    const std::vector<std::string_view> v(args);
    return parse_command_line(v);
}

bool exists_none(std::string_view) { return false; }
bool exists_all(std::string_view) { return true; }
bool exists_local_gguf(std::string_view p) { return p == "./local.gguf" || p == "/models/qwen.gguf" || p == "./my-engine.so"; }

std::string temp_dir(const char *name) {
    const auto dir = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir.string();
}

} // namespace

TEST_CASE("cli: parses commands, subcommands, flags and short aliases", "[cli]") {
    auto r = parse({"chat", "-m", "qwen", "--engine=llama", "--resume", "chat-1", "-y"});
    REQUIRE(r.ok);
    CHECK(r.value.command == "chat");
    CHECK(r.value.get("model") == "qwen");
    CHECK(r.value.get("engine") == "llama");
    CHECK(r.value.get("resume") == "chat-1");
    CHECK(r.value.has("yes"));
    CHECK_FALSE(r.value.has("list"));
    CHECK(r.value.get("ctx", "4096") == "4096");

    auto pull = parse({"--config-dir", "/tmp/cfg", "model", "pull", "Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M", "--name", "qwen"});
    REQUIRE(pull.ok);
    CHECK(pull.value.command == "model");
    CHECK(pull.value.subcommand == "pull");
    REQUIRE(pull.value.positionals.size() == 1);
    CHECK(pull.value.positionals[0] == "Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M");
    CHECK(pull.value.get("config-dir") == "/tmp/cfg");
    CHECK(pull.value.get("name") == "qwen");

    auto add = parse({"engine", "add", "--", "-weird-name.so"});
    REQUIRE(add.ok);
    CHECK(add.value.positionals[0] == "-weird-name.so");

    auto run = parse({"run", "graph.json"});
    REQUIRE(run.ok);
    CHECK(run.value.positionals[0] == "graph.json");

    auto help = parse({"chat", "--help"});
    REQUIRE(help.ok);
    CHECK(help.value.help);

    auto none = parse({});
    REQUIRE(none.ok);
    CHECK(none.value.command.empty());
}

TEST_CASE("cli: rejects bad command lines", "[cli]") {
    CHECK_FALSE(parse({"frobnicate"}).ok);
    CHECK_FALSE(parse({"engine"}).ok);                // missing subcommand
    CHECK_FALSE(parse({"engine", "list"}).ok);        // unknown subcommand
    CHECK_FALSE(parse({"engine", "add"}).ok);         // missing plugin path
    CHECK_FALSE(parse({"engine", "ls", "extra"}).ok); // too many arguments
    CHECK_FALSE(parse({"chat", "--model"}).ok);       // flag without value
    CHECK_FALSE(parse({"chat", "--name", "x"}).ok);   // flag of another command
    CHECK_FALSE(parse({"chat", "--yes=1"}).ok);       // bool flag with a value
    CHECK_FALSE(parse({"chat", "-z"}).ok);            // unknown short option
    CHECK_FALSE(parse({"run"}).ok);                   // missing graph file
    auto e = parse({"model", "pull", "a", "--bogus"});
    REQUIRE_FALSE(e.ok);
    CHECK(e.error.message.find("--bogus") != std::string::npos);
}

TEST_CASE("cli: config.json round trip", "[cli]") {
    CliConfig c;
    c.workspace = "/home/me/agent ws";
    c.models_dir = "C:\\Users\\me\\models";
    c.default_model = "qwen2.5-0.5b-instruct-q4_k_m";
    c.capabilities = {"llama", "audiocpp"};
    c.backend = "vulkan";
    c.hardware.cpu_cores = 8;
    c.hardware.nvidia_smi = false;
    c.hardware.nvidia_device = true;
    c.hardware.dri_nodes = {"/dev/dri/renderD128"};
    c.context_window = 8192;
    c.max_tokens = 256;
    c.temperature = 0.25f;
    c.memory_budget_mb = 6144;

    auto back = config_from_json(config_to_json(c));
    REQUIRE(back.ok);
    CHECK(back.value.workspace == c.workspace);
    CHECK(back.value.models_dir == c.models_dir);
    CHECK(back.value.default_model == c.default_model);
    CHECK(back.value.capabilities == c.capabilities);
    CHECK(back.value.backend == "vulkan");
    CHECK(back.value.hardware.cpu_cores == 8);
    CHECK(back.value.hardware.nvidia_device);
    CHECK_FALSE(back.value.hardware.nvidia_smi);
    CHECK(back.value.hardware.dri_nodes == c.hardware.dri_nodes);
    CHECK(back.value.context_window == 8192);
    CHECK(back.value.max_tokens == 256);
    CHECK(back.value.temperature == Catch::Approx(0.25f));
    CHECK(back.value.memory_budget_mb == 6144);

    // Unknown keys are ignored; missing keys keep their defaults.
    auto partial = config_from_json(R"({"workspace": "/w", "future_key": {"x": 1}, "temperature": 1})");
    REQUIRE(partial.ok);
    CHECK(partial.value.workspace == "/w");
    CHECK(partial.value.context_window == 4096);
    CHECK(partial.value.temperature == Catch::Approx(1.0f));
    CHECK(partial.value.memory_budget_mb == 0);

    CHECK_FALSE(config_from_json("[1,2]").ok);
    CHECK_FALSE(config_from_json("{\"context_window\": 0}").ok);
    CHECK_FALSE(config_from_json("not json").ok);
    CHECK_FALSE(config_from_json("{\"memory_budget_mb\": -1}").ok);
}

TEST_CASE("cli: config and engines files on disk", "[cli]") {
    const std::string dir = temp_dir("agent_test_cli_config");
    const std::string cfg_path = dir + "/sub/config.json";
    const std::string eng_path = dir + "/sub/engines.json";

    auto missing = load_config(cfg_path);
    REQUIRE(missing.ok);
    CHECK(missing.value.workspace.empty());
    auto no_engines = load_engines(eng_path);
    REQUIRE(no_engines.ok);
    CHECK(no_engines.value.empty());

    CliConfig c;
    c.default_model = "tiny";
    REQUIRE(save_config(cfg_path, c).ok); // creates sub/
    auto loaded = load_config(cfg_path);
    REQUIRE(loaded.ok);
    CHECK(loaded.value.default_model == "tiny");
    CHECK_FALSE(std::filesystem::exists(cfg_path + ".tmp"));

    std::vector<EngineEntry> engines = {{"llama", "/opt/agent/agent-engine-llama.so", "b12020", {"text_gen", "tool_call"}, "local"}};
    REQUIRE(save_engines(eng_path, engines).ok);
    auto le = load_engines(eng_path);
    REQUIRE(le.ok);
    REQUIRE(le.value.size() == 1);
    CHECK(le.value[0].name == "llama");
    CHECK(le.value[0].caps == engines[0].caps);
    CHECK(le.value[0].backend == "local");
}

TEST_CASE("cli: engines.json round trip and edits", "[cli]") {
    std::vector<EngineEntry> engines;
    upsert_engine(engines, {"llama", "/a/agent-engine-llama.so", "1", {"text_gen"}, "cpu"});
    upsert_engine(engines, {"sdcpp", "/a/agent-engine-sdcpp.so", "2", {"image_gen", "image_edit"}, "cuda"});
    upsert_engine(engines, {"llama", "/b/agent-engine-llama.so", "3", {"text_gen", "vision_in"}, "vulkan"});
    REQUIRE(engines.size() == 2);
    CHECK(engines[0].path == "/b/agent-engine-llama.so");

    auto back = engines_from_json(engines_to_json(engines));
    REQUIRE(back.ok);
    REQUIRE(back.value.size() == 2);
    CHECK(back.value[0].name == "llama");
    CHECK(back.value[0].version == "3");
    CHECK(back.value[0].backend == "vulkan");
    CHECK(back.value[1].caps == std::vector<std::string>{"image_gen", "image_edit"});

    CHECK(find_engine_entry(back.value, "sdcpp") != nullptr);
    CHECK(remove_engine(back.value, "sdcpp"));
    CHECK_FALSE(remove_engine(back.value, "sdcpp"));
    CHECK(find_engine_entry(back.value, "sdcpp") == nullptr);

    auto empty = engines_from_json(engines_to_json({}));
    REQUIRE(empty.ok);
    CHECK(empty.value.empty());

    // The minimal manifest agent::load_engine_manifest accepts: the name comes from the file name.
    auto minimal = engines_from_json(R"({"engines": [{"path": "plugins/agent-engine-onnx.so"}]})");
    REQUIRE(minimal.ok);
    CHECK(minimal.value[0].name == "onnx");
    CHECK(minimal.value[0].backend == "cpu");

    CHECK_FALSE(engines_from_json(R"({"engines": [{"name": "x"}]})").ok); // no path
    CHECK_FALSE(engines_from_json(R"({"plugins": []})").ok);
}

TEST_CASE("cli: engines run out of process unless --inprocess or engines.json say otherwise", "[cli]") {
    CHECK(cli_engine_isolation(false, "") == agent::EngineIsolation::Process);
    CHECK(cli_engine_isolation(false, "process") == agent::EngineIsolation::Process);
    CHECK(cli_engine_isolation(false, "inprocess") == agent::EngineIsolation::InProcess);
    CHECK(cli_engine_isolation(true, "process") == agent::EngineIsolation::InProcess);
    // GPU device nodes are opened in the engine sandbox only for GPU backends.
    CHECK_FALSE(cli_engine_gpu("cpu"));
    CHECK_FALSE(cli_engine_gpu(""));
    CHECK(cli_engine_gpu("cuda"));
    CHECK(cli_engine_gpu("vulkan"));

    EngineEntry trusted{"llama", "/a/agent-engine-llama.so", "1", {"text_gen"}, "cpu"};
    trusted.isolation = "inprocess";
    const std::vector<EngineEntry> engines = {trusted};
    auto back = engines_from_json(engines_to_json(engines));
    REQUIRE(back.ok);
    CHECK(back.value[0].isolation == "inprocess");
    CHECK_FALSE(engines_from_json(R"({"engines": [{"path": "a.so", "isolation": "maybe"}]})").ok);

    auto chat = parse_command_line(std::vector<std::string_view>{"chat", "--inprocess"});
    REQUIRE(chat.ok);
    CHECK(chat.value.has("inprocess"));
    auto run = parse_command_line(std::vector<std::string_view>{"run", "g.json", "--inprocess"});
    REQUIRE(run.ok);
    CHECK(run.value.has("inprocess"));
}

TEST_CASE("cli: model spec resolution", "[cli]") {
    const std::vector<ModelRef> catalog = {
        {"stories260K", "/models/stories260K.gguf"}, {"sd-turbo", "/models/sd-turbo.safetensors"}, {"qwen", "/models/qwen.gguf"}};

    auto by_name = resolve_model("qwen", {}, catalog, exists_all);
    REQUIRE(by_name.ok);
    CHECK(by_name.value.path == "/models/qwen.gguf");

    auto by_path = resolve_model("./local.gguf", {}, catalog, exists_local_gguf);
    REQUIRE(by_path.ok);
    CHECK(by_path.value.path == "./local.gguf");
    CHECK(by_path.value.name == "local");

    auto by_default = resolve_model("", "qwen", catalog, exists_all);
    REQUIRE(by_default.ok);
    CHECK(by_default.value.name == "qwen");

    auto first_chat = resolve_model("", "", catalog, exists_all);
    REQUIRE(first_chat.ok);
    CHECK(first_chat.value.name == "stories260K");

    auto missing_file = resolve_model("stories260K", {}, catalog, exists_none);
    REQUIRE_FALSE(missing_file.ok);
    CHECK(missing_file.error.message.find("missing") != std::string::npos);

    auto unknown = resolve_model("Qwen/Qwen3-GGUF:Q4_K_M", {}, catalog, exists_none);
    REQUIRE_FALSE(unknown.ok);
    CHECK(unknown.error.message.find("agent model pull") != std::string::npos);

    CHECK_FALSE(resolve_model("", "", {}, exists_none).ok);
}

TEST_CASE("cli: engine resolution by model type and --engine", "[cli]") {
    CHECK(engine_for_model("/m/Qwen2.5-0.5B-Q4_K_M.GGUF") == "llama");
    CHECK(engine_for_model("add.onnx") == "onnx");
    CHECK(engine_for_model("sd_xl_turbo.safetensors") == "sdcpp");
    CHECK(engine_for_model("model.ckpt") == "sdcpp");
    CHECK(engine_for_model("voices.bin").empty());

    const std::vector<EngineEntry> engines = {{"llama", "/e/agent-engine-llama.so", "", {}, "cpu"},
                                              {"sdcpp", "/e/agent-engine-sdcpp.so", "", {}, "cpu"}};

    auto by_type = resolve_engine("", "/models/qwen.gguf", engines, exists_none);
    REQUIRE(by_type.ok);
    CHECK(by_type.value.name == "llama");
    CHECK(by_type.value.path == "/e/agent-engine-llama.so");

    auto by_name = resolve_engine("sdcpp", "/models/qwen.gguf", engines, exists_none);
    REQUIRE(by_name.ok);
    CHECK(by_name.value.path == "/e/agent-engine-sdcpp.so");

    auto by_path = resolve_engine("./my-engine.so", "/models/qwen.gguf", engines, exists_local_gguf);
    REQUIRE(by_path.ok);
    CHECK(by_path.value.name.empty());
    CHECK(by_path.value.path == "./my-engine.so");

    // Known by type but not in engines.json: the caller searches the plugin directories.
    auto not_installed = resolve_engine("", "add.onnx", engines, exists_none);
    REQUIRE(not_installed.ok);
    CHECK(not_installed.value.name == "onnx");
    CHECK(not_installed.value.path.empty());

    CHECK_FALSE(resolve_engine("", "voices.bin", engines, exists_none).ok);
    CHECK_FALSE(resolve_engine("/no/such/agent-engine-x.so", "a.gguf", engines, exists_none).ok);
}

TEST_CASE("cli: slash command parsing", "[cli]") {
    auto plain = parse_slash_command("  hello there  ");
    CHECK(plain.kind == SlashKind::None);
    CHECK(plain.text == "hello there");

    auto escaped = parse_slash_command("//etc/passwd is a path");
    CHECK(escaped.kind == SlashKind::None);
    CHECK(escaped.text == "/etc/passwd is a path");

    CHECK(parse_slash_command("/exit").kind == SlashKind::Exit);
    CHECK(parse_slash_command("/QUIT").kind == SlashKind::Exit);
    CHECK(parse_slash_command("/clear").kind == SlashKind::Clear);
    CHECK(parse_slash_command("/stats").kind == SlashKind::Stats);
    CHECK(parse_slash_command("/help").kind == SlashKind::Help);

    auto model = parse_slash_command("/model qwen");
    CHECK(model.kind == SlashKind::Model);
    CHECK(model.arg == "qwen");

    auto image = parse_slash_command("/image cat.png what breed is this?");
    CHECK(image.kind == SlashKind::Image);
    CHECK(image.arg == "cat.png");
    CHECK(image.rest == "what breed is this?");

    auto quoted = parse_slash_command("/image \"my photos/cat 1.jpg\"   describe");
    CHECK(quoted.arg == "my photos/cat 1.jpg");
    CHECK(quoted.rest == "describe");

    auto open = parse_slash_command("/open 2");
    CHECK(open.kind == SlashKind::Open);
    CHECK(open.arg == "2");
    CHECK(parse_slash_command("/open").arg.empty());

    auto unknown = parse_slash_command("/frob x");
    CHECK(unknown.kind == SlashKind::Unknown);
    CHECK(unknown.arg == "frob");
}

TEST_CASE("cli: small helpers", "[cli]") {
    CHECK(json_escape("a\"b\\c\nd\x01") == "a\\\"b\\\\c\\nd\\u0001");
    CHECK(split_list(" llama, AudioCpp,,sdcpp llama ") == std::vector<std::string>{"llama", "audiocpp", "sdcpp"});
    CHECK(split_list("").empty());
    CHECK(caps_to_names((1u << 0) | (1u << 1) | (1u << 10)) == std::vector<std::string>{"text_gen", "tool_call", "image_gen"});
    CHECK(plugin_file_name("llama").rfind("agent-engine-llama.", 0) == 0);
    CHECK(mime_for_path("A.PNG") == "image/png");
    CHECK(mime_for_path("x.wav") == "audio/wav");
    CHECK(mime_for_path("x.unknown") == "application/octet-stream");
    CHECK(starter_pack().size() == 4);

    HardwareInfo hw;
    CHECK(suggest_backend(hw) == "cpu");
    hw.dri_nodes = {"/dev/dri/renderD128"};
    CHECK(suggest_backend(hw) == "vulkan");
    hw.nvidia_smi = true;
    CHECK(suggest_backend(hw) == "cuda");

    std::string index = add_conversation_id("", "a");
    index = add_conversation_id(index, "b");
    index = add_conversation_id(index, "a"); // moves to the end
    CHECK(conversation_ids(index) == std::vector<std::string>{"b", "a"});

    CHECK(format_progress(0, 0) == "0.0 MiB");
    const std::string half = format_progress(512 * 1024, 1024 * 1024, 10);
    CHECK(half.rfind("[#####-----]", 0) == 0);
    CHECK(half.find("50.0%") != std::string::npos);

    CHECK(one_line("a\nb", 10) == "a b");
    CHECK(one_line("abcdefgh", 4) == "abcd...");
    CHECK(one_line("ab\xc3\xa9z", 3) == "ab..."); // never splits the 2-byte 'e acute'
}

TEST_CASE("cli: stream filter hides tool-call markup, also across token boundaries", "[cli]") {
    StreamFilter f;
    std::string shown;
    for (std::string_view tok :
         {"Let me check.", "<tool", "_call>\n{\"name\": \"list_dir\"", ", \"arguments\": {}}\n</tool_", "call>\n", "Done <", "b>"})
        shown += filter_stream(f, tok);
    shown += flush_stream(f);
    CHECK(shown == "Let me check.Done <b>");

    StreamFilter g;
    CHECK(filter_stream(g, "a <tool_c") == "a ");
    CHECK(flush_stream(g) == "<tool_c"); // an unfinished prefix outside a call is printed at the end

    StreamFilter h;
    CHECK(filter_stream(h, "<tool_call>{\"name\": \"x\"").empty());
    CHECK(flush_stream(h).empty()); // an unterminated call is dropped
    CHECK(filter_stream(h, "plain") == "plain");
}

TEST_CASE("cli: turn footer shows prompt speed, generation speed and KV size", "[cli][footer]") {
    using agent::TokenUsage;
    using agent::cli::format_turn_footer;

    // An engine that reports nothing (speech, image, ...) gets no footer.
    CHECK(format_turn_footer(TokenUsage{.prompt_tokens = 10, .completion_tokens = 5}).empty());

    TokenUsage u;
    u.prompt_tokens = 412;
    u.cached_tokens = 380;
    u.completion_tokens = 64;
    u.prefill_ms = 100.0f; // 32 fresh tokens in 0.1 s
    u.decode_ms = 700.0f;  // 63 tokens after the first, in 0.7 s
    u.kv_tokens = 476;
    u.kv_max_tokens = 2048;
    u.kv_type = 4;
    u.kv_bytes = 3 * 1024 * 1024 + 100 * 1024;
    CHECK(format_turn_footer(u) == "prompt 412 tok (380 cached) 320 tok/s | gen 64 tok 90 tok/s | KV 476/2048 q4_0 3.1 MiB");

    // Without a cache hit nothing is claimed as cached, and a missing KV type or size is left out.
    TokenUsage cold;
    cold.prompt_tokens = 100;
    cold.completion_tokens = 11;
    cold.prefill_ms = 200.0f;
    cold.decode_ms = 100.0f;
    cold.kv_tokens = 111;
    cold.kv_max_tokens = 4096;
    CHECK(format_turn_footer(cold) == "prompt 100 tok 500 tok/s | gen 11 tok 100 tok/s | KV 111/4096");
}

TEST_CASE("cli: the usage text names every command main() dispatches", "[cli]") {
    const std::string usage = agent::cli::usage_all();
    for (const char *command : {"init", "doctor", "clean", "secret", "setup", "engine", "model", "chat", "run", "voice", "task", "version",
                                "runtime", "skill", "mcp", "sandbox"}) {
        INFO(command);
        CHECK(usage.find(std::string("  ") + command) != std::string::npos);
    }
    CHECK(usage.find("--download-engines") != std::string::npos);
    CHECK(usage.find("--install-runtimes") != std::string::npos);
}

TEST_CASE("--no-sandbox is a global flag that turns the sandboxes off", "[cli]") {
    agent::Config on;
    apply_sandbox_choice(on);
    CHECK(on.engine_sandbox.enabled);
    CHECK(on.tool_sandbox.enabled);
    const std::vector<std::string_view> args = {"doctor", "--no-sandbox"};
    auto cl = parse_command_line(args);
    REQUIRE(cl.ok);
    CHECK(cl.value.has("no-sandbox"));
    set_no_sandbox(true);
    agent::Config off;
    apply_sandbox_choice(off);
    CHECK_FALSE(off.engine_sandbox.enabled);
    CHECK_FALSE(off.tool_sandbox.enabled);
    set_no_sandbox(false);
}

#if defined(AGENT_CLI_HAS_TUI)
#include "tui.hpp"

TEST_CASE("tui: markdown-lite layout", "[cli][tui]") {
    const std::string md = "# Title\n\nSome **bold** and `code` words that are long enough to wrap around a narrow screen.\n\n- first "
                           "item\n- second item\n\n```\nint x = 1;\n```\n";
    const std::string text = render_markdown_text(md, 30);
    CHECK(text.find("Title") != std::string::npos);
    CHECK(text.find("bold") != std::string::npos);
    CHECK(text.find("**") == std::string::npos); // the markers are not shown
    CHECK(text.find("`") == std::string::npos);
    CHECK(text.find("first item") != std::string::npos);
    CHECK(text.find("int x = 1;") != std::string::npos);
    CHECK(text.find("narrow screen.") != std::string::npos); // the long sentence wrapped, none of it was cut off
    CHECK(text.find("```") == std::string::npos);
    // Wrapped to 30 columns: more than one line holds the long sentence.
    size_t lines = 0;
    for (char c : text) lines += c == '\n';
    CHECK(lines >= 8);
}
#endif

// ------------------------------------------------------------------ schedule: cron text

#include "schedule.hpp"

TEST_CASE("schedule: intervals become cron expressions", "[cli][schedule]") {
    CHECK(cron_from_every("15m").value == "*/15 * * * *");
    CHECK(cron_from_every("1m").value == "* * * * *");
    CHECK(cron_from_every("2h").value == "0 */2 * * *");
    CHECK(cron_from_every("1d").value == "0 0 * * *");
    CHECK_FALSE(cron_from_every("7m").ok); // does not divide 60: the runs would not be evenly spaced
    CHECK_FALSE(cron_from_every("5h").ok);
    CHECK_FALSE(cron_from_every("2d").ok);
    CHECK_FALSE(cron_from_every("m").ok);
    CHECK_FALSE(cron_from_every("15x").ok);
    CHECK(cron_from_daily("09:30").value == "30 9 * * *");
    CHECK(cron_from_daily("9:5").value == "5 9 * * *");
    CHECK_FALSE(cron_from_daily("25:00").ok);
    CHECK_FALSE(cron_from_daily("nine").ok);
    CHECK(check_cron("0  9 * * 1-5").value == "0 9 * * 1-5");
    CHECK_FALSE(check_cron("0 9 * *").ok);
    CHECK_FALSE(check_cron("0 9 * * mon").ok);
}

TEST_CASE("schedule: crontab entries are added, found and removed without touching other lines", "[cli][schedule]") {
    const std::string other = "MAILTO=me\n0 3 * * * /usr/bin/backup --all\n";
    const std::string one = crontab_block("nightly", "0 2 * * *", "/opt/agent/agent --home /h run g.json --yes");
    const std::string two = crontab_block("hourly", "0 * * * *", "/opt/agent/agent run 'my graph.json'");
    const std::string crontab = other + one + two;

    const auto entries = parse_crontab(crontab);
    REQUIRE(entries.size() == 2);
    CHECK(entries[0].name == "nightly");
    CHECK(entries[0].expression == "0 2 * * *");
    CHECK(entries[0].command == "/opt/agent/agent --home /h run g.json --yes");
    CHECK(entries[1].name == "hourly");
    CHECK(entries[1].command == "/opt/agent/agent run 'my graph.json'");

    CHECK(crontab_without(crontab, "nightly") == other + two);
    CHECK(crontab_without(crontab, "hourly") == other + one);
    CHECK(crontab_without(crontab, "missing") == crontab);
    CHECK(parse_crontab(crontab_without(crontab_without(crontab, "nightly"), "hourly")).empty());
    CHECK(crontab_without(crontab_without(crontab, "nightly"), "hourly") == other);
}

TEST_CASE("schedule: quoting and names", "[cli][schedule]") {
    CHECK(quote_posix("run") == "run");
    CHECK(quote_posix("my graph.json") == "'my graph.json'");
    CHECK(quote_posix("it's") == "'it'\\''s'");
    CHECK(quote_posix("") == "''");
    CHECK(quote_windows("run") == "run");
    CHECK(quote_windows("C:\\Program Files\\agent.exe") == "\"C:\\Program Files\\agent.exe\"");
    CHECK(quote_windows("say \"hi\"") == "\"say \\\"hi\\\"\"");
    CHECK(valid_schedule_name("nightly-report_2"));
    CHECK_FALSE(valid_schedule_name(""));
    CHECK_FALSE(valid_schedule_name("a b"));
    CHECK_FALSE(valid_schedule_name("../x"));
}

// ------------------------------------------------------------------ waiting runs (agent wake)

#include "waiting.hpp"

TEST_CASE("wake: the waiting list round-trips, replaces by thread and picks what is due", "[cli][wake]") {
    std::vector<WaitingRun> runs;
    waiting_upsert(runs, {"nightly-1", "/g/a.json", "/work", 2000, {"--model", "qwen", "--yes"}});
    waiting_upsert(runs, {"morning-7", "/g/b.json", "", 1000, {"--workspace", "/ws"}});
    CHECK(runs.size() == 2);
    waiting_upsert(runs, {"nightly-1", "/g/a.json", "/work", 3000, {"--model", "qwen"}}); // the same thread waits again
    REQUIRE(runs.size() == 2);
    CHECK(runs[0].until == 3000);

    const auto parsed = waiting_from_json(waiting_to_json(runs));
    REQUIRE(parsed.ok);
    REQUIRE(parsed.value.size() == 2);
    CHECK(parsed.value[0].thread == "nightly-1");
    CHECK(parsed.value[0].flags == std::vector<std::string>{"--model", "qwen"});
    CHECK(parsed.value[0].cwd == "/work");
    CHECK(parsed.value[1].graph == "/g/b.json");

    CHECK(waiting_due(runs, 999).empty());
    const auto due = waiting_due(runs, 3500);
    REQUIRE(due.size() == 2);
    CHECK(due[0].thread == "morning-7"); // the earliest first
    CHECK(waiting_due(runs, 1000).size() == 1);

    CHECK(waiting_resume_args(runs[1]) == std::vector<std::string>{"run", "/g/b.json", "--workspace", "/ws", "--resume", "morning-7"});

    CHECK(waiting_remove(runs, "nightly-1"));
    CHECK_FALSE(waiting_remove(runs, "nightly-1"));
    CHECK(runs.size() == 1);

    // Odd text survives; an empty file and a broken one are not an error to start from.
    std::vector<WaitingRun> odd;
    waiting_upsert(odd, {"t\"q", "C:\\g dir\\x.json", "C:\\work dir", 5, {"--input", "a \"b\""}});
    const auto again = waiting_from_json(waiting_to_json(odd));
    REQUIRE(again.ok);
    CHECK(again.value[0].thread == "t\"q");
    CHECK(again.value[0].graph == "C:\\g dir\\x.json");
    CHECK(waiting_from_json("").ok);
    CHECK_FALSE(waiting_from_json("{not json").ok);
}

// ------------------------------------------------------------------ slash commands that run subcommands

TEST_CASE("chat: /doctor and /setup map to agent subcommands", "[cli][slash]") {
    auto argv = [](const char *line) { return slash_cli_argv(parse_slash_command(line)); };
    CHECK(parse_slash_command("/doctor").kind == SlashKind::Cli);
    CHECK(argv("/doctor") == std::vector<std::string>{"doctor"});
    CHECK(argv("/setup") == std::vector<std::string>{"setup"}); // alone it asks its questions in the chat
    CHECK(argv("/setup --yes --caps llama") == std::vector<std::string>{"setup", "--yes", "--caps", "llama"});
    CHECK(argv("/setup --home \"my repo/x y\"") == std::vector<std::string>{"setup", "--home", "my repo/x y"});
    // Other commands are not forwarded.
    CHECK(argv("/model").empty());
    CHECK(argv("hello").empty());
    CHECK(parse_slash_command("/clear").kind == SlashKind::Clear);
}
