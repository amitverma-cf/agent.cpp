// The `agent` command line program run as a real process (AGENT_TEST_CLI_PATH) against the mock engine, in a private home folder.
// Covers the command dispatcher, the small commands and the chat / task / run paths end to end.
#if defined(AGENT_TEST_CLI_PATH) && defined(AGENT_TEST_MOCK_ENGINE_PATH)

#include "cli_e2e_util.hpp"

using namespace cli_e2e;

TEST_CASE("cli e2e: version, help and bad commands", "[cli_e2e]") {
    Home h;
    auto v = h.run("version");
    CHECK(v.code == 0);
    CHECK(has(v, "agent "));
    auto bad = h.run("no-such-command");
    CHECK(bad.code == 2);
    CHECK(has(bad, "unknown command"));
    CHECK(has(bad, "agent --help"));
    auto missing = h.run("model rm");
    CHECK(missing.code != 0);
}

TEST_CASE("cli e2e: the listing commands work in an empty home", "[cli_e2e]") {
    Home h;
    for (const char *cmd : {"doctor", "engine ls", "model ls", "sandbox ls", "mcp ls", "runtime ls", "skill ls"}) {
        auto r = h.run(cmd);
        INFO(cmd << ": " << r.out);
        CHECK(r.code == 0);
        if (std::string(cmd) != "skill ls") CHECK_FALSE(r.out.empty()); // an empty skill list prints nothing
    }
}

TEST_CASE("cli e2e: engine add, list and remove", "[cli_e2e]") {
    Home h;
    auto add = h.run("engine add " + quote(AGENT_TEST_MOCK_ENGINE_PATH));
    INFO(add.out);
    CHECK(add.code == 0);
    auto ls = h.run("engine ls");
    CHECK(has(ls, "mock"));
    CHECK(has(ls, "text_gen"));
    auto rm = h.run("engine rm mock");
    CHECK(rm.code == 0);
    CHECK(has(h.run("engine ls"), "no engines"));
    CHECK(h.run("engine rm mock").code != 0); // already gone
    CHECK(h.run("engine add " + quote((h.root / "not-a-plugin.so").string())).code != 0);
}

TEST_CASE("cli e2e: sandbox grants are stored, listed and revoked", "[cli_e2e]") {
    Home h;
    const std::string dir = quote(h.workspace.string());
    auto grant = h.run("sandbox grant " + dir + " --write --for run_command");
    INFO(grant.out);
    CHECK(grant.code == 0);
    auto ls = h.run("sandbox ls");
    CHECK(has(ls, "write"));
    CHECK(has(ls, "run_command"));
    CHECK(h.run("sandbox revoke " + dir).code == 0);
    CHECK(has(h.run("sandbox ls"), "nothing granted"));
    CHECK(h.run("sandbox grant").code != 0); // a path is required
}

TEST_CASE("cli e2e: task runs one engine call", "[cli_e2e]") {
    Home h;
    auto r = h.run("task text_gen " + h.engine_args() + " --text hello");
    INFO(r.out);
    CHECK(r.code == 0);
    CHECK(has(r, "task: text_gen"));
    CHECK(has(r, "echo: hello"));
    CHECK(has(r, "usage:"));
    auto isolated = h.run("task text_gen -e " + quote(AGENT_TEST_MOCK_ENGINE_PATH) + " -m " + quote(h.model.string()) +
                          " --text again --workspace " + quote(h.workspace.string())); // out of process, through the engine host
    INFO(isolated.out);
    CHECK(has(isolated, "echo: again"));
    CHECK(h.run("task not_a_task " + h.engine_args()).code != 0);
    CHECK(h.run("task text_gen -e " + quote(AGENT_TEST_MOCK_ENGINE_PATH) + " -m " + quote((h.root / "missing.gguf").string())).code != 0);
}

TEST_CASE("cli e2e: chat answers, saves the conversation and resumes it", "[cli_e2e]") {
    Home h;
    auto r = h.run("chat " + h.engine_args(), "hi there\n/help\n/exit\n");
    INFO(r.out);
    CHECK(r.code == 0);
    CHECK(has(r, "echo: hi there"));
    CHECK(has(r, "saved as chat-"));
    auto list = h.run("chat --list " + h.engine_args());
    CHECK(has(list, "chat-"));
    const auto at = r.out.find("--resume ");
    REQUIRE(at != std::string::npos);
    const std::string id = r.out.substr(at + 9, r.out.find_first_of(" \r\n)", at + 9) - (at + 9));
    auto again = h.run("chat --resume " + id + " " + h.engine_args(), "second message\n/exit\n");
    INFO(again.out);
    CHECK(has(again, "echo: second message"));
}

TEST_CASE("cli e2e: run executes a graph file", "[cli_e2e]") {
    Home h;
    auto r = h.run("run " + quote(h.graph.string()) + " " + h.engine_args() + " --set request=ping");
    INFO(r.out);
    CHECK(r.code == 0);
    CHECK(has(r, "status: completed"));
    CHECK(has(r, "answer = echo: ping"));
    CHECK(h.run("run " + quote((h.root / "nope.json").string()) + " " + h.engine_args()).code != 0);
}

TEST_CASE("cli e2e: init creates the project folders", "[cli_e2e]") {
    Home h;
    const fs::path proj = h.root / "proj";
    auto r = h.run("init " + quote(proj.string()));
    INFO(r.out);
    CHECK(r.code == 0);
    CHECK(fs::exists(proj / ".agents"));
    CHECK(fs::exists(proj / ".agents" / "memory"));
    CHECK(h.run("init " + quote(proj.string())).code == 0); // running it again is harmless
}

TEST_CASE("cli e2e: runtime, skill and secret commands report usage errors and missing things", "[cli_e2e]") {
    Home h;
    CHECK(h.run("runtime").code == 2); // a subcommand is required
    CHECK(has(h.run("runtime ls"), "uv"));
    CHECK(h.run("runtime install").code == 2);
    CHECK(h.run("runtime install nonsense").code == 2);
    CHECK(h.run("runtime use ruby").code == 2);
    auto path = h.run("runtime path uv");
    CHECK(path.code == 1);
    CHECK(has(path, "not installed"));
    CHECK(h.run("skill rm ../escape").code == 2);
    CHECK(h.run("skill rm no-such-skill").code == 1);
    CHECK(h.run("skill").code == 2);
    CHECK(h.run("mcp rm nothing").code == 1);
    CHECK(h.run("mcp").code == 2);
    CHECK(h.run("secret").code == 2);
    CHECK(h.run("secret set").code != 0);
    CHECK(h.run("secret bind ONLYNAME").code != 0);
}

TEST_CASE("cli e2e: setup writes the home, doctor reads it, clean has nothing to do", "[cli_e2e]") {
    Home h;
    auto setup = h.run("setup --yes --no-models --caps llama --home " + quote((h.root / "home2").string()));
    INFO(setup.out);
    CHECK(setup.code == 0);
    CHECK(fs::exists(h.root / "home2" / "config.json"));
    auto doctor = h.run("doctor --home " + quote((h.root / "home2").string()));
    CHECK(doctor.code == 0);
    CHECK(has(doctor, "home:"));
    CHECK(has(doctor, "engines:"));
    for (const char *what : {"--downloads", "--cache", "--logs"}) {
        auto clean = h.run(std::string("clean ") + what + " --yes");
        INFO(what << ": " << clean.out);
        CHECK(clean.code == 0);
    }
    CHECK(h.run("clean").code != 0); // says what to clean
}

TEST_CASE("cli e2e: model commands that need no network", "[cli_e2e]") {
    Home h;
    auto search = h.run("model search qwen");
    CHECK(search.code == 0);
    CHECK(has(search, "qwen"));
    CHECK(h.run("model search zzzz-no-such-model").code == 0); // an empty table
    CHECK(h.run("model rm nope").code != 0);
    CHECK(h.run("model pull").code != 0);
    CHECK(h.run("model info").code != 0);
}

TEST_CASE("cli e2e: task and chat error paths and chat slash commands", "[cli_e2e]") {
    Home h;
    // The mock engine has no embedding or OCR capability.
    auto embed = h.run("task embed " + h.engine_args() + " --text hello");
    CHECK(embed.code != 0);
    CHECK_FALSE(embed.out.empty());

    auto chat = h.run("chat " + h.engine_args(), "/help\n/stats\n/clear\nafter clear\n/model\n/open\n/image\n/nonsense\n/exit\n");
    INFO(chat.out);
    CHECK(chat.code == 0);
    CHECK(has(chat, "echo: after clear"));
    CHECK(has(chat, "saved as chat-"));
    auto bad_ctx = h.run("chat " + h.engine_args() + " --ctx notanumber", "/exit\n");
    CHECK(bad_ctx.code != 0);
    auto opts = h.run("chat " + h.engine_args() + " --max-tokens 16 --ctx 2048 --engine-options \"{}\"", "hello\n/exit\n");
    CHECK(has(opts, "echo: hello"));
    auto bad_json = h.run("chat " + h.engine_args() + " --engine-options \"{bad\"", "/exit\n");
    CHECK(bad_json.code != 0);
}

TEST_CASE("cli e2e: run with a thread id, --json, --input and a resume", "[cli_e2e]") {
    Home h;
    auto first = h.run("run " + quote(h.graph.string()) + " " + h.engine_args() + " --thread t1 --set request=one --json");
    INFO(first.out);
    CHECK(first.code == 0);
    CHECK(has(first, "echo: one"));
    CHECK(has(first, "{"));
    auto input = h.run("run " + quote(h.graph.string()) + " " + h.engine_args() +
                       " --thread t3 --input typed-input"); // --input answers an interrupted run: nothing to answer here
    CHECK(input.code == 0);
    auto resume = h.run("run " + quote(h.graph.string()) + " " + h.engine_args() + " --resume t1");
    CHECK_FALSE(resume.out.empty());
    std::ofstream(h.root / "broken.json") << "{not json";
    CHECK(h.run("run " + quote((h.root / "broken.json").string()) + " " + h.engine_args()).code != 0);
    CHECK(h.run("run").code != 0);
}

TEST_CASE("cli e2e: global options choose the home", "[cli_e2e]") {
    Home h;
    const std::string home = quote((h.root / "chosen").string());
    CHECK(h.run("engine ls --home " + home).code == 0);
    CHECK(h.run("engine ls --config-dir " + quote((h.root / "oldlayout").string())).code == 0);
    auto help = h.run("--help");
    CHECK(has(help, "usage: agent"));
    CHECK(h.run("-h").code == 0);
}

#if defined(AGENT_TEST_MOCK_ALL_ENGINE_PATH)
TEST_CASE("cli e2e: task drives every capability of the mock-all engine", "[cli_e2e][mock_all]") {
    Home h;
    const std::string audio = (h.root / "in.wav").string(), image = (h.root / "in.png").string();
    std::ofstream(audio, std::ios::binary) << "RIFFxxxx";
    std::ofstream(image, std::ios::binary) << "\x89PNG";
    const std::string eng = std::string(" -e ") + quote(AGENT_TEST_MOCK_ALL_ENGINE_PATH) + " -m " + quote(h.model.string()) +
                            " --inprocess --workspace " + quote(h.workspace.string());
    auto task = [&](const std::string &args) { return h.run("task " + args + eng); };
    auto expect = [&](const std::string &args, const std::string &needle, int code = 0) {
        auto r = task(args);
        INFO(args << " -> " << r.out);
        CHECK(r.code == code);
        CHECK(has(r, needle));
    };
    expect("asr --file " + quote(audio), "mock transcript");
    expect("asr", "needs an audio part", 1);
    expect("ocr --file " + quote(image), "mock ocr text");
    expect("tts --text hi", "audio/wav");
    expect("vad --file " + quote(audio), "segments");
    expect("diarize --file " + quote(audio), "speaker");
    expect("image_gen --text cat", "image/png");
    expect("video_gen --text cat", "video/mp4");
    expect("mesh_gen --file " + quote(image), "model/gltf-binary");
    expect("embed --text hello", "embedding");
    expect("rerank --text \"red fox\" --doc \"a red fox\" --doc blue", "score[0]");
    expect("decide --text q --params \"{\\\"mode\\\":\\\"yesno\\\"}\"", "\"yes\"");
    expect("classify --text q --params \"{\\\"mode\\\":\\\"yesno\\\"}\"", "probs");
    expect("generic --tensor x:f32:2=1,2", "tensor: x");
    expect("ocr --text mock-error", "simulated engine failure", 1);
    expect("ocr --text mock-empty", "task: ocr");
    expect("ocr --text mock-slow", "slow task finished");
    // Out of process, through the engine host.
    auto isolated = h.run("task embed -e " + quote(AGENT_TEST_MOCK_ALL_ENGINE_PATH) + " -m " + quote(h.model.string()) +
                          " --text hello --workspace " + quote(h.workspace.string()));
    CHECK(has(isolated, "embedding"));
}
#endif

#if defined(__linux__)
// The encrypted file store (AGENT_SECRETS_PASSPHRASE): the OS credential stores of Windows and macOS are not touched by tests.
TEST_CASE("cli e2e: secrets are stored, bound to a consumer, unbound and removed", "[cli_e2e][secrets]") {
    Home h;
    set_env("AGENT_SECRETS_PASSPHRASE", "test passphrase");
    auto set = h.run("secret set MYKEY --stdin", "sk-test-123\n");
    INFO(set.out);
    CHECK(set.code == 0);
    CHECK_FALSE(has(set, "sk-test-123")); // the value is never printed
    auto ls = h.run("secret ls");
    CHECK(has(ls, "MYKEY"));
    CHECK(has(ls, "not bound"));
    CHECK(has(h.run("secret bind MYKEY tool:web_fetch"), "tool:web_fetch"));
    CHECK(has(h.run("secret ls"), "usable by: tool:web_fetch"));
    CHECK(h.run("secret bind MYKEY nonsense").code != 0);
    CHECK(h.run("secret bind NOPE tool:x").code != 0);
    CHECK(h.run("secret set \"bad name\" --stdin", "x\n").code != 0);
    CHECK(h.run("secret set EMPTY --stdin", "\n").code != 0);
    CHECK(h.run("secret unbind MYKEY tool:web_fetch").code == 0);
    CHECK(has(h.run("secret ls"), "not bound"));
    CHECK(h.run("secret rm MYKEY").code == 0);
    CHECK(has(h.run("secret ls"), "no secrets"));
    set_env("AGENT_SECRETS_PASSPHRASE", "");
}
#endif

#endif
