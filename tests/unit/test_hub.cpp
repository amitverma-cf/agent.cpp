#include <agent-cpp/hub.hpp>
#include <catch_amalgamated.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace agent::hub;

namespace {

void write_file(const std::string &path, std::string_view text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

constexpr std::string_view kRepoJson = R"({
  "id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
  "gated": false,
  "tags": ["gguf", "license:apache-2.0"],
  "cardData": {"license": "apache-2.0", "language": ["en"]},
  "siblings": [
    {"rfilename": ".gitattributes", "size": 1561},
    {"rfilename": "README.md", "size": 4856},
    {"rfilename": "qwen2.5-0.5b-instruct-q4_k_m.gguf", "size": 491400032,
     "lfs": {"sha256": "6EB923E7D26E9CEA28811E1A8E852009B21242FB157B26149D3B188F3A8C8653", "size": 491400032, "pointerSize": 135}},
    {"rfilename": "qwen2.5-0.5b-instruct-q4_k_s.gguf", "size": 100},
    {"rfilename": "qwen2.5-0.5b-instruct-q4_k.gguf", "size": 90},
    {"rfilename": "qwen2.5-0.5b-instruct-q8_0.gguf", "size": 200},
    {"rfilename": "big/model-q5_k_m-00001-of-00002.gguf", "size": 10},
    {"rfilename": "big/model-q5_k_m-00002-of-00002.gguf", "size": 10},
    {"rfilename": "mmproj-q8_0.gguf", "size": 5}
  ]
})";

} // namespace

TEST_CASE("parse_spec takes the links copied from a model page", "[hub]") {
    auto file = parse_spec("https://huggingface.co/o/r/blob/main/dir/file.gguf?download=true");
    REQUIRE(file.ok);
    CHECK(file.value.repo == "o/r");
    CHECK(file.value.revision == "main");
    CHECK(file.value.file == "dir/file.gguf");
    auto direct = parse_spec("https://huggingface.co/o/r/resolve/v1/m.onnx");
    REQUIRE(direct.ok);
    CHECK(direct.value.revision == "v1");
    CHECK(direct.value.file == "m.onnx");
    auto page = parse_spec("https://huggingface.co/o/r");
    REQUIRE(page.ok);
    CHECK(page.value.repo == "o/r");
    CHECK(page.value.file.empty());
    auto tree = parse_spec("huggingface.co/o/r/tree/main");
    REQUIRE(tree.ok);
    CHECK(tree.value.repo == "o/r");
}

TEST_CASE("parse_spec handles file, quant, bare repo and revisions", "[hub]") {
    auto a = parse_spec("Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m.gguf");
    REQUIRE(a.ok);
    CHECK(a.value.repo == "Qwen/Qwen2.5-0.5B-Instruct-GGUF");
    CHECK(a.value.file == "qwen2.5-0.5b-instruct-q4_k_m.gguf");
    CHECK(a.value.quant.empty());
    CHECK(a.value.revision == "main");

    auto b = parse_spec("  bartowski/Foo-GGUF:Q4_K_M ");
    REQUIRE(b.ok);
    CHECK(b.value.repo == "bartowski/Foo-GGUF");
    CHECK(b.value.quant == "Q4_K_M");
    CHECK(b.value.file.empty());

    auto c = parse_spec("https://huggingface.co/ggml-org/models@v2:tinyllamas/stories260K.gguf");
    REQUIRE(c.ok);
    CHECK(c.value.repo == "ggml-org/models");
    CHECK(c.value.revision == "v2");
    CHECK(c.value.file == "tinyllamas/stories260K.gguf");

    auto d = parse_spec("hf.co/owner/name");
    REQUIRE(d.ok);
    CHECK(d.value.repo == "owner/name");
    CHECK(d.value.file.empty());
    CHECK(d.value.quant.empty());

    CHECK_FALSE(parse_spec("").ok);
    CHECK_FALSE(parse_spec("noslash").ok);
    CHECK_FALSE(parse_spec("a/b/c").ok);
    CHECK_FALSE(parse_spec("owner/repo:").ok);
    CHECK_FALSE(parse_spec("own er/repo").ok);
    CHECK_FALSE(parse_spec("owner/repo:../../etc/passwd.gguf").ok);
    CHECK_FALSE(parse_spec("owner/repo:/abs/x.gguf").ok);
    CHECK_FALSE(parse_spec("owner/repo:dir\\x.gguf").ok);
}

TEST_CASE("quant_matches is case-insensitive and filename-based", "[hub]") {
    CHECK(quant_matches("qwen2.5-0.5b-instruct-q4_k_m.gguf", "Q4_K_M"));
    CHECK(quant_matches("Model-IQ2_XXS.gguf", "iq2_xxs"));
    CHECK_FALSE(quant_matches("model-q8_0.gguf", "Q4_K_M"));
    CHECK_FALSE(quant_matches("Q4_K_M/model-f16.gguf", "Q4_K_M")); // directory name does not count
    CHECK_FALSE(quant_matches("model.gguf", ""));
}

TEST_CASE("parse_repo_info and select_file", "[hub]") {
    auto info = parse_repo_info(kRepoJson);
    REQUIRE(info.ok);
    CHECK(info.value.repo == "Qwen/Qwen2.5-0.5B-Instruct-GGUF");
    CHECK(info.value.license == "apache-2.0");
    CHECK_FALSE(info.value.gated);
    REQUIRE(info.value.files.size() == 9);

    ModelSpec spec{.repo = "Qwen/Qwen2.5-0.5B-Instruct-GGUF", .quant = "q4_k_m"};
    auto f = select_file(info.value, spec);
    REQUIRE(f.ok);
    CHECK(f.value.path == "qwen2.5-0.5b-instruct-q4_k_m.gguf");
    CHECK(f.value.size == 491400032);
    CHECK(f.value.sha256 == "6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653");

    // Q4_K prefers the exact tag over Q4_K_M / Q4_K_S.
    spec.quant = "Q4_K";
    f = select_file(info.value, spec);
    REQUIRE(f.ok);
    CHECK(f.value.path == "qwen2.5-0.5b-instruct-q4_k.gguf");

    // Q8_0 must not pick the mmproj file.
    spec.quant = "Q8_0";
    f = select_file(info.value, spec);
    REQUIRE(f.ok);
    CHECK(f.value.path == "qwen2.5-0.5b-instruct-q8_0.gguf");

    spec.quant = "Q5_K_M"; // split GGUF only: the first shard is selected, split_set expands it
    f = select_file(info.value, spec);
    REQUIRE(f.ok);
    CHECK(f.value.path == "big/model-q5_k_m-00001-of-00002.gguf");
    auto set = split_set(info.value, f.value);
    REQUIRE(set.ok);
    REQUIRE(set.value.size() == 2);
    CHECK(set.value[1].path == "big/model-q5_k_m-00002-of-00002.gguf");

    // A plain file is its own set; a set with a missing shard fails.
    auto single = split_set(info.value, RemoteFile{.path = "qwen2.5-0.5b-instruct-q8_0.gguf"});
    REQUIRE(single.ok);
    CHECK(single.value.size() == 1);
    RepoInfo partial = info.value;
    std::erase_if(partial.files, [](const RemoteFile &rf) { return rf.path == "big/model-q5_k_m-00002-of-00002.gguf"; });
    CHECK(split_set(partial, f.value).error.code == agent::ErrorCode::KeyNotFound);

    spec.quant = "Q2_K";
    CHECK(select_file(info.value, spec).error.code == agent::ErrorCode::KeyNotFound);

    spec.quant.clear(); // several ggufs, no selector -> ambiguous
    CHECK_FALSE(select_file(info.value, spec).ok);

    spec.file = "qwen2.5-0.5b-instruct-q8_0.gguf";
    f = select_file(info.value, spec);
    REQUIRE(f.ok);
    CHECK(f.value.size == 200);

    spec.file = "model-q5_k_m-00001-of-00002.gguf"; // unique basename
    f = select_file(info.value, spec);
    REQUIRE(f.ok);
    CHECK(f.value.path == "big/model-q5_k_m-00001-of-00002.gguf");

    spec.file = "missing.gguf";
    CHECK(select_file(info.value, spec).error.code == agent::ErrorCode::KeyNotFound);
}

TEST_CASE("parse_repo_info licence fallbacks", "[hub]") {
    auto tags = parse_repo_info(R"({"id":"a/b","tags":["gguf","license:cc-by-nc-4.0"],"siblings":[]})");
    REQUIRE(tags.ok);
    CHECK(tags.value.license == "cc-by-nc-4.0");

    auto other =
        parse_repo_info(R"({"id":"a/b","gated":"manual","cardData":{"license":"other","license_name":"fish-audio-research-license"}})");
    REQUIRE(other.ok);
    CHECK(other.value.gated);
    CHECK(other.value.license == "other");
    CHECK(other.value.license_name == "fish-audio-research-license");

    CHECK_FALSE(parse_repo_info("not json").ok);
}

TEST_CASE("licence policy denies non-commercial and research licences", "[hub]") {
    for (const char *ok_license : {"apache-2.0", "mit", "bsd-3-clause", "cc-by-4.0", "cc-by-sa-4.0", "llama3.1", "gemma", "openrail", ""})
        CHECK_FALSE(is_restricted_license(ok_license));
    for (const char *bad : {"cc-by-nc-4.0", "CC-BY-NC-SA-4.0", "cc-by-nc-nd-4.0", "deepfloyd-if-license", "apple-amlr"})
        CHECK(is_restricted_license(bad));
    CHECK(is_restricted_license("other", "fish-audio-research-license"));
    CHECK(is_restricted_license("other", "Qwen Research License"));
    CHECK(is_restricted_license("other", "Some Non-Commercial License"));
    CHECK_FALSE(is_restricted_license("other", "tongyi-qianwen"));
}

TEST_CASE("sha256 known vectors", "[hub]") {
    CHECK(sha256_hex(std::string_view("")) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(sha256_hex(std::string_view("abc")) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(sha256_hex(std::string_view("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    // One million 'a' through the streaming file path (crosses the 1 MiB read buffer).
    const std::string path = (std::filesystem::temp_directory_path() / "agent_hub_sha_test.bin").string();
    write_file(path, std::string(1000000, 'a'));
    auto digest = sha256_file(path);
    REQUIRE(digest.ok);
    CHECK(digest.value == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    std::filesystem::remove(path);
    CHECK_FALSE(sha256_file(path).ok);
}

TEST_CASE("models.json round trip, escaping and remove", "[hub]") {
    const auto dir = std::filesystem::temp_directory_path() / "agent_hub_catalog_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string catalog = (dir / "models.json").string();

    auto empty = load_catalog(catalog);
    REQUIRE(empty.ok);
    CHECK(empty.value.empty());

    const std::string model_path = (dir / "m.gguf").string();
    write_file(model_path, "GGUF");
    std::vector<CatalogEntry> entries = {
        {.name = "qwen", .repo = "Qwen/X", .file = "x.gguf", .path = model_path, .size = 4, .sha256 = "ab", .license = "apache-2.0"},
        {.name = "we\"ird\\name\n\t\x01",
         .repo = "a/b",
         .file = "dir/y.gguf",
         .path = "C:\\models\\y.gguf",
         .size = 18446744073709551615ull,
         .sha256 = "",
         .license = "other (custom)"},
    };
    REQUIRE(save_catalog(catalog, entries).ok);
    CHECK_FALSE(std::filesystem::exists(catalog + ".tmp"));

    auto loaded = load_catalog(catalog);
    REQUIRE(loaded.ok);
    REQUIRE(loaded.value.size() == 2);
    for (size_t i = 0; i < entries.size(); ++i) {
        CHECK(loaded.value[i].name == entries[i].name);
        CHECK(loaded.value[i].repo == entries[i].repo);
        CHECK(loaded.value[i].file == entries[i].file);
        CHECK(loaded.value[i].path == entries[i].path);
        CHECK(loaded.value[i].size == entries[i].size);
        CHECK(loaded.value[i].sha256 == entries[i].sha256);
        CHECK(loaded.value[i].license == entries[i].license);
    }

    CHECK(remove_model("nope", catalog).error.code == agent::ErrorCode::KeyNotFound);
    REQUIRE(remove_model("qwen", catalog).ok);
    CHECK_FALSE(std::filesystem::exists(model_path));
    auto after = list_models(catalog);
    REQUIRE(after.ok);
    REQUIRE(after.value.size() == 1);
    CHECK(after.value[0].repo == "a/b");

    CHECK(catalog_to_json({}) == "{\n  \"models\": []\n}\n");
    CHECK_FALSE(catalog_from_json("{\"models\": [{\"repo\": \"a/b\"}]}").ok); // entry without name
    CHECK_FALSE(catalog_from_json("[]").ok);
    std::filesystem::remove_all(dir);
}

TEST_CASE("model registry specs from models.json entries", "[hub][models]") {
    const std::string dir = "/tmp/agent_test_hub_specs";
    std::filesystem::create_directories(dir);
    write_file(dir + "/chat.gguf", "gguf");
    const std::vector<CatalogEntry> entries = {
        {"chat", "o/r", "chat.gguf", dir + "/chat.gguf", 4, "", "apache-2.0"},
        {"vision", "o/r", "v.onnx", dir + "/missing.onnx", 1234, "", ""},
        {"sd", "o/r", "sd.SAFETENSORS", dir + "/sd.SAFETENSORS", 9, "", ""},
    };
    agent::Config base;
    base.workspace_dir = "/tmp/ws";
    base.context_window = 8192;
    const std::vector<agent::ModelSpec> specs = model_specs_from_catalog(entries, base);
    REQUIRE(specs.size() == 3);
    CHECK(specs[0].name == "chat");
    CHECK(specs[0].config.provider == agent::AiProvider::Engine);
    CHECK(specs[0].config.engine == "llama");
    CHECK(specs[0].config.model == dir + "/chat.gguf");
    CHECK(specs[0].config.context_window == 8192);
    CHECK(specs[0].config.workspace_dir == "/tmp/ws");
    CHECK(specs[0].estimated_bytes == 0); // file present: the registry estimates it
    CHECK(specs[1].config.engine == "onnx");
    CHECK(specs[1].estimated_bytes == 1234); // file missing: catalogued size
    CHECK(specs[2].config.engine == "sdcpp");

    base.engine = "/opt/agent-engine-custom.so"; // an explicit engine wins
    CHECK(model_spec_from_entry(entries[0], base).config.engine == "/opt/agent-engine-custom.so");
    std::filesystem::remove_all(dir);
}

TEST_CASE("proxy URL parsing and NO_PROXY matching", "[hub]") {
    auto p = parse_proxy_url("http://127.0.0.1:39125");
    REQUIRE(p.ok);
    CHECK(p.value.host == "127.0.0.1");
    CHECK(p.value.port == 39125);

    p = parse_proxy_url("http://user:p%40ss@proxy.corp:3128/");
    REQUIRE(p.ok);
    CHECK(p.value.host == "proxy.corp");
    CHECK(p.value.port == 3128);

    p = parse_proxy_url("proxy.corp");
    REQUIRE(p.ok);
    CHECK(p.value.port == 80);

    p = parse_proxy_url("http://[::1]:8080");
    REQUIRE(p.ok);
    CHECK(p.value.host == "::1");
    CHECK(p.value.port == 8080);

    CHECK_FALSE(parse_proxy_url("").ok);
    CHECK_FALSE(parse_proxy_url("http://host:99999").ok);

    const std::string_view no_proxy = "localhost,127.0.0.1, .internal.net,example.com,*.svc.local";
    CHECK(host_bypasses_proxy("localhost", no_proxy));
    CHECK(host_bypasses_proxy("a.internal.net", no_proxy));
    CHECK(host_bypasses_proxy("internal.net", no_proxy));
    CHECK(host_bypasses_proxy("cdn.example.com", no_proxy));
    CHECK(host_bypasses_proxy("x.svc.local", no_proxy));
    CHECK_FALSE(host_bypasses_proxy("notexample.com", no_proxy));
    CHECK_FALSE(host_bypasses_proxy("huggingface.co", no_proxy));
    CHECK(host_bypasses_proxy("anything", "*"));
    CHECK_FALSE(host_bypasses_proxy("huggingface.co", ""));
}

TEST_CASE("proxy URL credentials are split and percent-decoded", "[hub]") {
    auto p = parse_proxy_url("http://alice:p%40ss%3Aw@proxy.corp:3128/");
    REQUIRE(p.ok);
    CHECK(p.value.host == "proxy.corp");
    CHECK(p.value.port == 3128);
    CHECK(p.value.user == "alice");
    CHECK(p.value.pass == "p@ss:w");

    p = parse_proxy_url("bob@proxy.corp:8080");
    REQUIRE(p.ok);
    CHECK(p.value.user == "bob");
    CHECK(p.value.pass.empty());

    p = parse_proxy_url("http://proxy.corp:8080");
    REQUIRE(p.ok);
    CHECK(p.value.user.empty());
    CHECK_FALSE(parse_proxy_url("http://user:pw@").ok);
}

TEST_CASE("NO_PROXY matches host suffixes on label boundaries only", "[hub]") {
    CHECK(host_bypasses_proxy("HuggingFace.co", "huggingface.co"));
    CHECK(host_bypasses_proxy("cdn-lfs.huggingface.co", "huggingface.co"));
    CHECK_FALSE(host_bypasses_proxy("evilhuggingface.co", "huggingface.co"));
    CHECK_FALSE(host_bypasses_proxy("huggingface.co.evil.net", "huggingface.co"));
    CHECK(host_bypasses_proxy("a.b.corp", " , .corp ,"));
}

TEST_CASE("HF token source precedence", "[hub]") {
    CHECK(resolve_hf_token("explicit", "env", "legacy", "file") == "explicit");
    CHECK(resolve_hf_token("", "env", "legacy", "file") == "env");
    CHECK(resolve_hf_token("", "", "legacy", "file") == "legacy");
    CHECK(resolve_hf_token("", "", "", "file\n") == "file");
    CHECK(resolve_hf_token("  ", " \n", "", "  tok \r\n") == "tok");
    CHECK(resolve_hf_token("", "", "", "").empty());
}

TEST_CASE("HF token is only sent to Hugging Face hosts", "[hub]") {
    const std::string_view hf = "https://huggingface.co";
    CHECK(token_allowed_for_host("huggingface.co", hf));
    CHECK(token_allowed_for_host("HuggingFace.co", hf));
    CHECK(token_allowed_for_host("hf.co", hf));
    // Pre-signed CDN and third-party redirect targets never get the token.
    CHECK_FALSE(token_allowed_for_host("cdn-lfs.huggingface.co", hf));
    CHECK_FALSE(token_allowed_for_host("cas-bridge.xethub.hf.co", hf));
    CHECK_FALSE(token_allowed_for_host("evilhuggingface.co", hf));
    CHECK_FALSE(token_allowed_for_host("huggingface.co.evil.net", hf));
    CHECK_FALSE(token_allowed_for_host("s3.amazonaws.com", hf));
    CHECK_FALSE(token_allowed_for_host("", hf));

    // A mirror endpoint: only the mirror host, not Hugging Face.
    CHECK(token_allowed_for_host("hf-mirror.example", "https://hf-mirror.example:8443/"));
    CHECK_FALSE(token_allowed_for_host("huggingface.co", "https://hf-mirror.example"));
    CHECK_FALSE(token_allowed_for_host("hf.co", "https://hf-mirror.example"));
}

// Network test, opt-in: AGENT_TEST_HUB_NETWORK=1. Uses HTTPS_PROXY / SSL_CERT_FILE etc. from the env.
TEST_CASE("hub downloads stories260K from Hugging Face", "[hub][network]") {
    const char *flag = std::getenv("AGENT_TEST_HUB_NETWORK");
    if (!flag || std::string_view(flag) != "1") { SKIP("set AGENT_TEST_HUB_NETWORK=1 to run"); }
    const auto dir = std::filesystem::temp_directory_path() / "agent_hub_network_test";
    std::filesystem::remove_all(dir);

    const HttpOptions http = http_options_from_env();

    // Raw download with a cross-host redirect (huggingface.co -> CDN), no checksum.
    const std::string raw = (dir / "raw" / "stories260K.gguf").string();
    auto dl = download_file(http.endpoint + "/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf", raw, http, {});
    INFO(dl.error.message);
    REQUIRE(dl.ok);
    CHECK(std::filesystem::file_size(raw) == 1185376);
    CHECK_FALSE(std::filesystem::exists(raw + ".part"));

    // Resume: keep the first 100000 bytes as a .part file and pull through the catalogue.
    const std::string catalog = (dir / "models.json").string();
    const std::string expected = (dir / "ggml-org" / "models" / "tinyllamas" / "stories260K.gguf").string();
    std::filesystem::create_directories(std::filesystem::path(expected).parent_path());
    {
        std::ifstream in(raw, std::ios::binary);
        std::string head(100000, '\0');
        in.read(head.data(), static_cast<std::streamsize>(head.size()));
        write_file(expected + ".part", head);
    }
    struct Progress {
        uint64_t calls = 0, last = 0, total = 0;
    } progress;
    PullOptions opts;
    opts.models_dir = dir.string();
    opts.catalog_path = catalog;
    opts.http = http;
    opts.progress_user_data = &progress;
    opts.on_progress = [](uint64_t done, uint64_t total, void *user) {
        auto *p = static_cast<Progress *>(user);
        p->calls++;
        p->last = done;
        p->total = total;
        return true;
    };
    auto pulled = pull_model("ggml-org/models:tinyllamas/stories260K.gguf", opts);
    INFO(pulled.error.message);
    REQUIRE(pulled.ok);
    CHECK(pulled.value.name == "stories260K");
    CHECK(pulled.value.size == 1185376);
    CHECK(pulled.value.sha256 == "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d");
    CHECK(std::filesystem::file_size(pulled.value.path) == 1185376);
    CHECK(progress.calls > 0);
    CHECK(progress.last == 1185376);
    CHECK(progress.total == 1185376);

    auto listed = list_models(catalog);
    REQUIRE(listed.ok);
    REQUIRE(listed.value.size() == 1);
    CHECK(listed.value[0].path == pulled.value.path);

    REQUIRE(remove_model("stories260K", catalog).ok);
    CHECK_FALSE(std::filesystem::exists(pulled.value.path));
    std::filesystem::remove_all(dir);
}

TEST_CASE("select_mmproj picks the projector that goes with the model file", "[hub][mmproj]") {
    auto info = parse_repo_info(R"({"id":"x/vlm","siblings":[
        {"rfilename":"vlm-Q8_0.gguf","size":100},{"rfilename":"vlm-f16.gguf","size":200},
        {"rfilename":"mmproj-vlm-Q8_0.gguf","size":10},{"rfilename":"mmproj-vlm-f16.gguf","size":20}]})");
    REQUIRE(info.ok);

    auto same_quant = select_mmproj(info.value, "vlm-f16.gguf");
    REQUIRE(same_quant.ok);
    REQUIRE(same_quant.value.has_value());
    CHECK(same_quant.value->path == "mmproj-vlm-f16.gguf");

    auto fallback = select_mmproj(info.value, "vlm-Q4_K_M.gguf"); // no Q4 projector: Q8_0 is the default
    REQUIRE(fallback.ok);
    CHECK(fallback.value->path == "mmproj-vlm-Q8_0.gguf");

    auto wanted = select_mmproj(info.value, "vlm-Q8_0.gguf", "f16");
    REQUIRE(wanted.ok);
    CHECK(wanted.value->path == "mmproj-vlm-f16.gguf");

    CHECK_FALSE(select_mmproj(info.value, "vlm-Q8_0.gguf", "none").value.has_value());
    CHECK_FALSE(select_mmproj(info.value, "mmproj-vlm-Q8_0.gguf").value.has_value()); // already a projector
    CHECK_FALSE(select_mmproj(info.value, "vlm-Q8_0.gguf", "bf16").ok);               // asked for one that is not there

    auto text_only = parse_repo_info(R"({"id":"x/llm","siblings":[{"rfilename":"llm-Q4_K_M.gguf","size":1}]})");
    REQUIRE(text_only.ok);
    CHECK_FALSE(select_mmproj(text_only.value, "llm-Q4_K_M.gguf").value.has_value());
}

TEST_CASE("with_mmproj_option adds the projector to engine options once", "[hub][mmproj]") {
    CHECK(with_mmproj_option("", "/m/p.gguf") == R"({"mmproj":"/m/p.gguf"})");
    CHECK(with_mmproj_option("{}", "/m/p.gguf") == R"({"mmproj":"/m/p.gguf"})");
    CHECK(with_mmproj_option(R"({"threads":4})", "/m/p.gguf") == R"({"mmproj":"/m/p.gguf","threads":4})");
    CHECK(with_mmproj_option(R"({"mmproj":"other.gguf"})", "/m/p.gguf") == R"({"mmproj":"other.gguf"})"); // the user's wins
    CHECK(with_mmproj_option(R"({"threads":4})", "") == R"({"threads":4})");
    CHECK(with_mmproj_option("", R"(C:\models\p.gguf)") == R"({"mmproj":"C:\\models\\p.gguf"})"); // Windows paths stay valid JSON
}

TEST_CASE("the catalogue keeps the projector of a model", "[hub][mmproj]") {
    CatalogEntry e{.name = "vlm",
                   .repo = "x/vlm",
                   .file = "vlm.gguf",
                   .path = "/m/vlm.gguf",
                   .size = 5,
                   .license = "mit",
                   .mmproj = "/m/mmproj.gguf",
                   .mmproj_file = "mmproj.gguf"};
    auto back = catalog_from_json(catalog_to_json(std::span<const CatalogEntry>(&e, 1)));
    REQUIRE(back.ok);
    REQUIRE(back.value.size() == 1);
    CHECK(back.value[0].mmproj == "/m/mmproj.gguf");
    CHECK(back.value[0].mmproj_file == "mmproj.gguf");
    CatalogEntry plain{.name = "llm", .repo = "x/llm", .file = "llm.gguf", .path = "/m/llm.gguf"};
    CHECK(catalog_to_json(std::span<const CatalogEntry>(&plain, 1)).find("mmproj") == std::string::npos); // old format unchanged
}
