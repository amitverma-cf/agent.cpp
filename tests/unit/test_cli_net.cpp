// The command line's network commands (model pull, engine pull, skills, MCP registry, runtimes) against a stand-in for Hugging Face,
// GitHub, skills.sh and the MCP registry on the loopback interface. HF_ENDPOINT and AGENT_HTTP_REWRITE point the real `agent` binary
// at it. Test code only: the product opens no listening socket.
#if defined(AGENT_TEST_CLI_PATH) && defined(AGENT_TEST_MOCK_ENGINE_PATH)

#include "cli_e2e_util.hpp"
#include "engine_release.hpp"
#include "runtimes.hpp"

#include <agent-cpp/hub.hpp>
#include <httplib.h>
#include <map>
#include <thread>

using namespace cli_e2e;

namespace {

std::string slurp(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string json_str(const std::string &s) { return "\"" + s + "\""; }

struct FakeNet {
    httplib::Server svr;
    std::thread th;
    int port = 0;
    std::map<std::string, std::string> hf_files; // "<repo>/<file>" -> bytes
    size_t manifest_size = 0;
    std::string requests;                         // every request path and status, for failure messages
    std::string engine_asset_bytes, manifest_sha; // the manifest lists the checksum of the bytes as first served

    std::string base() const { return "http://127.0.0.1:" + std::to_string(port); }

    // sha256 of the bytes, for the repo listing.
    static std::string sha(const std::string &bytes) {
        return agent::hub::sha256_hex(std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
    }

    std::string repo_json(const std::string &repo, const std::string &license, const std::vector<std::string> &files) const {
        std::string j = "{\"id\":" + json_str(repo) + ",\"cardData\":{\"license\":" + json_str(license) + "},\"siblings\":[";
        for (size_t i = 0; i < files.size(); ++i) {
            const std::string &bytes = hf_files.at(repo + "/" + files[i]);
            j += std::string(i ? "," : "") + "{\"rfilename\":" + json_str(files[i]) + ",\"size\":" + std::to_string(bytes.size()) +
                 ",\"lfs\":{\"sha256\":" + json_str(sha(bytes)) + ",\"size\":" + std::to_string(bytes.size()) + "}}";
        }
        return j + "]}";
    }

    FakeNet() {
        // ---- Hugging Face
        hf_files["acme/tiny-GGUF/tiny-Q4_K_M.gguf"] = std::string(3000, 'q');
        hf_files["acme/tiny-GGUF/tiny-Q8_0.gguf"] = std::string(5000, 'r');
        hf_files["acme/tiny-GGUF/mmproj-F16.gguf"] = std::string(1000, 'm');
        hf_files["acme/gated-GGUF/gated-Q4_K_M.gguf"] = std::string(2000, 'g');
        hf_files["acme/folder/a.bin"] = "AAAA";
        hf_files["acme/folder/sub/b.txt"] = "BBBB";
        svr.Get(R"(/api/models/acme/tiny-GGUF)", [this](const httplib::Request &, httplib::Response &res) {
            res.set_content(repo_json("acme/tiny-GGUF", "apache-2.0", {"tiny-Q4_K_M.gguf", "tiny-Q8_0.gguf", "mmproj-F16.gguf"}),
                            "application/json");
        });
        svr.Get(R"(/api/models/acme/gated-GGUF)", [this](const httplib::Request &, httplib::Response &res) {
            res.set_content(repo_json("acme/gated-GGUF", "cc-by-nc-4.0", {"gated-Q4_K_M.gguf"}), "application/json");
        });
        svr.Get(R"(/api/models/acme/folder)", [this](const httplib::Request &, httplib::Response &res) {
            res.set_content(repo_json("acme/folder", "mit", {"a.bin", "sub/b.txt"}), "application/json");
        });
        svr.Get(R"(/(acme/[^/]+)/resolve/main/(.+))", [this](const httplib::Request &req, httplib::Response &res) {
            auto it = hf_files.find(req.matches[1].str() + "/" + req.matches[2].str());
            if (it == hf_files.end()) {
                res.status = 404;
                return;
            }
            res.set_content(it->second, "application/octet-stream");
        });
        // ---- GitHub release (engine pull)
        svr.Get(R"(/gh/acme/agent.cpp/releases/download/v9.9.9/manifest.json)", [this](const httplib::Request &, httplib::Response &res) {
            const std::string asset = agent::cli::engine_asset_name("mock", agent::cli::host_os(), agent::cli::host_arch());
            res.set_content("{\"version\":\"v9.9.9\",\"repository\":\"acme/agent.cpp\",\"files\":[{\"name\":" + json_str(asset) +
                                ",\"sha256\":" + json_str(manifest_sha) + ",\"size\":" + std::to_string(manifest_size) + "}]}",
                            "application/json");
        });
        svr.Get(
            R"(/gh/acme/agent.cpp/releases/download/v9.9.9/(agent-engine-.+))",
            [this](const httplib::Request &, httplib::Response &res) { res.set_content(engine_asset_bytes, "application/octet-stream"); });
        // ---- runtimes: an asset that is not an archive
        svr.Get(R"(/gh/astral-sh/uv/releases/.*)",
                [](const httplib::Request &, httplib::Response &res) { res.set_content("not an archive", "application/octet-stream"); });
        // ---- skills.sh and GitHub's API
        svr.Get(R"(/skills/api/search)", [](const httplib::Request &, httplib::Response &res) {
            res.set_content(R"({"skills":[{"id":"acme/skills/pdf","source":"acme/skills","skillId":"pdf","name":"pdf","installs":42}]})",
                            "application/json");
        });
        svr.Get(R"(/ghapi/repos/acme/skills/git/trees/HEAD)", [](const httplib::Request &, httplib::Response &res) {
            res.set_content(
                R"({"truncated":false,"tree":[{"type":"blob","path":"pdf/SKILL.md","size":40},{"type":"blob","path":"pdf/scripts/run.py","size":12},{"type":"blob","path":"other/SKILL.md","size":5}]})",
                "application/json");
        });
        svr.Get(R"(/raw/acme/skills/HEAD/pdf/SKILL.md)", [](const httplib::Request &, httplib::Response &res) {
            res.set_content("---\nname: pdf\ndescription: test skill\n---\nbody\n", "text/plain");
        });
        svr.Get(R"(/raw/acme/skills/HEAD/pdf/scripts/run.py)",
                [](const httplib::Request &, httplib::Response &res) { res.set_content("print(1)\n", "text/plain"); });
        // ---- MCP registry
        svr.Get(R"(/mcp/v0.1/servers)", [](const httplib::Request &, httplib::Response &res) {
            res.set_content(
                R"({"servers":[{"server":{"name":"io.acme/files","description":"File tools","version":"1.2.3","repository":{"url":"https://github.com/acme/files"},)"
                R"("packages":[{"registryType":"npm","identifier":"@acme/files","version":"1.2.3","transport":{"type":"stdio"},)"
                R"("environmentVariables":[{"name":"API_KEY","isSecret":true,"isRequired":true},{"name":"MODE","default":"fast"}]}]}},)"
                R"({"server":{"name":"io.acme/hosted","description":"Hosted only","version":"1","remotes":[{"url":"https://example.invalid/mcp"}]}}]})",
                "application/json");
        });
        svr.set_logger([this](const httplib::Request &req, const httplib::Response &res) {
            requests += req.path + " -> " + std::to_string(res.status) + "\n";
        });
        port = svr.bind_to_any_port("127.0.0.1");
        th = std::thread([this] { svr.listen_after_bind(); });
        svr.wait_until_ready();
        const std::string b = base();
        set_env("HF_ENDPOINT", b);
        set_env("AGENT_HTTP_REWRITE", "https://github.com=" + b + "/gh;https://api.github.com=" + b +
                                          "/ghapi;https://raw.githubusercontent.com=" + b + "/raw;https://skills.sh=" + b +
                                          "/skills;https://registry.modelcontextprotocol.io=" + b + "/mcp");
        engine_asset_bytes = slurp(AGENT_TEST_MOCK_ENGINE_PATH);
        manifest_sha = sha(engine_asset_bytes);
        manifest_size = engine_asset_bytes.size();
    }
    ~FakeNet() {
        set_env("HF_ENDPOINT", "");
        set_env("AGENT_HTTP_REWRITE", "");
        svr.stop();
        if (th.joinable()) th.join();
    }
};

} // namespace

TEST_CASE("cli net: model pull, list, info and remove against a stand-in hub", "[cli_net]") {
    Home h;
    FakeNet net;
    auto info = h.run("model info acme/tiny-GGUF");
    INFO(info.out);
    CHECK(info.code == 0);
    CHECK(has(info, "tiny-Q4_K_M.gguf"));
    CHECK(has(info, "apache-2.0"));

    auto pull = h.run("model pull acme/tiny-GGUF:Q4_K_M --name tiny");
    INFO(pull.out);
    CHECK(pull.code == 0);
    auto ls = h.run("model ls");
    CHECK(has(ls, "tiny"));
    CHECK(has(ls, "acme/tiny-GGUF"));
    CHECK(has(ls, "mmproj")); // the projector was fetched with it

    CHECK(h.run("model rm tiny").code == 0);
    CHECK(has(h.run("model ls"), "no models"));

    auto missing = h.run("model pull acme/does-not-exist:Q4_K_M");
    CHECK(missing.code != 0);
    auto no_quant = h.run("model pull acme/tiny-GGUF:Q2_K");
    CHECK(no_quant.code != 0);

    auto folder = h.run("model pull acme/folder:*");
    INFO(folder.out);
    CHECK(folder.code == 0);
    CHECK(h.run("model pull acme/tiny-GGUF:Q8_0 --name big --no-mmproj").code == 0);
}

TEST_CASE("cli net: a repo with a restricted licence needs an explicit yes", "[cli_net]") {
    Home h;
    FakeNet net;
    auto refused = h.run("model pull acme/gated-GGUF:Q4_K_M");
    INFO(refused.out);
    CHECK(refused.code != 0);
    auto allowed = h.run("model pull acme/gated-GGUF:Q4_K_M --allow-restricted");
    CHECK(allowed.code == 0);
}

TEST_CASE("cli net: engine pull checks the manifest and the checksum", "[cli_net]") {
    Home h;
    FakeNet net;
    auto pull = h.run("engine pull mock --repo acme/agent.cpp --tag v9.9.9");
    INFO(pull.out << "\nserver saw:\n" << net.requests);
    CHECK(pull.code == 0);
    CHECK(has(h.run("engine ls"), "mock"));
    // An engine the release does not have, and a tampered download.
    auto other = h.run("engine pull nothere --repo acme/agent.cpp --tag v9.9.9");
    CHECK(other.code != 0);
    net.engine_asset_bytes += "tampered"; // the manifest still lists the checksum of the original bytes
    CHECK(h.run("engine pull mock --repo acme/agent.cpp --tag v9.9.9").code != 0);
    CHECK(h.run("engine pull ../evil --repo acme/agent.cpp --tag v9.9.9").code != 0); // invalid name
    auto bad_repo = h.run("engine pull mock --repo acme/other --tag v9.9.9");
    CHECK(bad_repo.code != 0);
}

TEST_CASE("cli net: skills are searched, installed, listed and removed", "[cli_net]") {
    Home h;
    FakeNet net;
    auto search = h.run("skill search pdf");
    INFO(search.out);
    CHECK(search.code == 0);
    CHECK(has(search, "acme/skills/pdf"));
    auto add = h.run("skill add acme/skills/pdf --yes");
    INFO(add.out);
    CHECK(add.code == 0);
    CHECK(has(add, "scripts")); // the install notes that it brings a script
    CHECK(has(h.run("skill ls"), "pdf"));
    CHECK(h.run("skill add not-an-id --yes").code != 0);
    CHECK(h.run("skill add acme/skills/missing --yes").code != 0);
    CHECK(h.run("skill rm pdf").code == 0);
    CHECK_FALSE(has(h.run("skill ls"), "pdf"));
}

TEST_CASE("cli net: MCP servers are searched, added and removed", "[cli_net]") {
    Home h;
    FakeNet net;
    auto search = h.run("mcp search files");
    INFO(search.out.substr(0, 300) << "\nserver saw:\n" << net.requests);
    CHECK(search.code == 0);
    CHECK(has(search, "io.acme/files"));
    CHECK(has(search, "hosted")); // listed, though it cannot start here
    auto add = h.run("mcp add io.acme/files --yes");
    INFO(add.out);
    CHECK(add.code == 0);
    CHECK(has(add, "api_key")); // the secret it needs
    CHECK(has(h.run("mcp ls"), "files"));
    CHECK(h.run("mcp add io.acme/hosted --yes").code != 0);  // no package that runs here
    CHECK(h.run("mcp add io.acme/unknown --yes").code != 0); // not an exact registry name
    CHECK(h.run("mcp rm files").code == 0);
    CHECK(h.run("mcp rm files").code != 0);
}

TEST_CASE("cli net: a runtime download that is not an archive fails cleanly", "[cli_net]") {
    Home h;
    FakeNet net;
    auto r = h.run("runtime install uv");
    INFO(r.out);
    CHECK(r.code != 0);
    CHECK(h.run("runtime path uv").code == 1);
}

#endif
