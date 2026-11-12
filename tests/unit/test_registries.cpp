#include "registries.hpp"

#include <catch_amalgamated.hpp>

using namespace agent::cli;

namespace {

// Trimmed from real answers of skills.sh and the MCP registry.
const char *kSkillSearch = R"({"query":"pdf","searchType":"fuzzy","skills":[
 {"id":"anthropics/skills/pdf","source":"anthropics/skills","skillId":"pdf","name":"pdf","installs":226623},
 {"id":"openai/skills/pdf","source":"openai/skills","skillId":"pdf","name":"pdf","installs":13754},
 {"source":"x/y","name":"no id"}]})";

const char *kTree = R"({"sha":"abc","truncated":false,"tree":[
 {"path":"README.md","type":"blob","size":10},
 {"path":"skills","type":"tree"},
 {"path":"skills/pdf/SKILL.md","type":"blob","size":100},
 {"path":"skills/pdf/scripts/extract.py","type":"blob","size":200},
 {"path":"skills/pdf/reference/forms.md","type":"blob","size":50},
 {"path":"skills/pdfx/SKILL.md","type":"blob","size":70},
 {"path":"examples/pdf/SKILL.md","type":"blob","size":5},
 {"path":"examples/pdf/deep/note.txt","type":"blob","size":5}]})";

const char *kMcpSearch = R"({"servers":[
 {"server":{"name":"com.pulsemcp/remote-filesystem","description":"Remote files.","repository":{"url":"https://github.com/pulsemcp/mcp-servers"},"version":"0.1.2",
   "packages":[{"registryType":"npm","identifier":"remote-filesystem-mcp-server","version":"0.1.2","transport":{"type":"stdio"}}]},
  "_meta":{"io.modelcontextprotocol.registry/official":{"isLatest":false}}},
 {"server":{"name":"com.pulsemcp/remote-filesystem","description":"Remote files.","version":"0.1.3",
   "packages":[{"registryType":"npm","identifier":"remote-filesystem-mcp-server","version":"0.1.3","transport":{"type":"stdio"},
     "packageArguments":[{"type":"positional","value":"--readonly"}],
     "environmentVariables":[{"name":"GCS_BUCKET","description":"Bucket","isRequired":true},
                             {"name":"GCS_PRIVATE_KEY","description":"Key","isSecret":true},
                             {"name":"GCS_MAKE_PUBLIC","default":"false"},
                             {"name":"GCS_ROOT_PATH","description":"optional"}]}]},
  "_meta":{"io.modelcontextprotocol.registry/official":{"isLatest":true}}},
 {"server":{"name":"io.example/fetch","description":"Fetch.","version":"1.0.0",
   "packages":[{"registryType":"pypi","identifier":"mcp-server-fetch","version":"1.0.0","transport":{"type":"stdio"}}]}},
 {"server":{"name":"io.example/hosted","description":"Only hosted.","version":"2.0.0","remotes":[{"type":"streamable-http","url":"https://mcp.example.com"}]}},
 {"server":{"name":"io.example/docker","description":"Container.","version":"1.0.0",
   "packages":[{"registryType":"oci","identifier":"ghcr.io/x/y","version":"1","transport":{"type":"stdio"}}]}}]})";

} // namespace

TEST_CASE("registries: skills.sh search results", "[registries]") {
    CHECK(skill_search_url("pdf reader") == "https://skills.sh/api/search?q=pdf%20reader");
    auto r = parse_skill_search(kSkillSearch);
    REQUIRE(r.ok);
    REQUIRE(r.value.size() == 2); // the entry without an id is dropped
    CHECK(r.value[0].id == "anthropics/skills/pdf");
    CHECK(r.value[0].source == "anthropics/skills");
    CHECK(r.value[0].skill_id == "pdf");
    CHECK(r.value[0].installs == 226623);
    CHECK_FALSE(parse_skill_search("not json").ok);
    CHECK_FALSE(parse_skill_search(R"({"x":1})").ok);
}

TEST_CASE("registries: skill ids", "[registries]") {
    auto ok1 = parse_skill_id("anthropics/skills/pdf");
    REQUIRE(ok1.ok);
    CHECK(ok1.value.source == "anthropics/skills");
    CHECK(ok1.value.skill_id == "pdf");
    CHECK_FALSE(parse_skill_id("pdf").ok);
    CHECK_FALSE(parse_skill_id("owner/repo").ok);
    CHECK_FALSE(parse_skill_id("owner/repo/").ok);
    CHECK_FALSE(parse_skill_id("/repo/skill").ok);
    CHECK_FALSE(parse_skill_id("owner/repo/../x").ok);
}

TEST_CASE("registries: finding a skill folder in a repository listing", "[registries]") {
    auto r = find_skill_in_tree(kTree, "pdf");
    REQUIRE(r.ok);
    CHECK(r.value.dir == "skills/pdf"); // the shortest of the two folders named pdf
    CHECK(r.value.files.size() == 3);

    // Sibling folders with a similar name do not leak into the result.
    auto skills_only = find_skill_in_tree(R"({"truncated":false,"tree":[
        {"path":"skills/pdf/SKILL.md","type":"blob","size":100},
        {"path":"skills/pdf/scripts/extract.py","type":"blob","size":200},
        {"path":"skills/pdfx/SKILL.md","type":"blob","size":70}]})",
                                          "pdf");
    REQUIRE(skills_only.ok);
    CHECK(skills_only.value.dir == "skills/pdf");
    CHECK(skills_only.value.files == std::vector<std::string>{"skills/pdf/SKILL.md", "skills/pdf/scripts/extract.py"});
    CHECK(skills_only.value.bytes == 300);

    CHECK_FALSE(find_skill_in_tree(kTree, "missing").ok);
    CHECK_FALSE(find_skill_in_tree(R"({"truncated":true,"tree":[]})", "pdf").ok);

    // A skill at the root of its repository.
    auto root = find_skill_in_tree(
        R"({"truncated":false,"tree":[{"path":"SKILL.md","type":"blob","size":1},{"path":"ref.md","type":"blob","size":2}]})", "my-skill");
    REQUIRE(root.ok);
    CHECK(root.value.dir.empty());
    CHECK(root.value.files.size() == 2);
}

TEST_CASE("registries: MCP registry search keeps the latest version of each server", "[registries]") {
    CHECK(mcp_search_url("file system", 5) == "https://registry.modelcontextprotocol.io/v0.1/servers?search=file%20system&limit=5");
    auto r = parse_mcp_search(kMcpSearch);
    REQUIRE(r.ok);
    REQUIRE(r.value.size() == 4);
    CHECK(r.value[0].name == "com.pulsemcp/remote-filesystem");
    CHECK(r.value[0].version == "0.1.3");
    CHECK(r.value[1].packages[0].type == "pypi");
    CHECK(r.value[2].remotes == std::vector<std::string>{"https://mcp.example.com"});
    CHECK_FALSE(parse_mcp_search("{}").ok);
}

TEST_CASE("registries: a registry entry becomes a config with secrets by reference", "[registries]") {
    auto all = parse_mcp_search(kMcpSearch);
    REQUIRE(all.ok);

    auto e = mcp_entry_for(all.value[0]);
    REQUIRE(e.ok);
    CHECK(e.value.server_name == "remote-filesystem");
    CHECK(e.value.secrets == std::vector<std::string>{"gcs_private_key"});
    CHECK(e.value.to_fill == std::vector<std::string>{"GCS_BUCKET"});
    CHECK(e.value.json.find("\"command\": \"npx\"") != std::string::npos);
    CHECK(e.value.json.find("\"-y\", \"remote-filesystem-mcp-server@0.1.3\", \"--readonly\"") != std::string::npos);
    CHECK(e.value.json.find("\"GCS_PRIVATE_KEY\": \"${secret:gcs_private_key}\"") != std::string::npos);
    CHECK(e.value.json.find("\"GCS_MAKE_PUBLIC\": \"false\"") != std::string::npos);
    CHECK(e.value.json.find("GCS_ROOT_PATH") == std::string::npos); // optional, no default

    auto pypi = mcp_entry_for(all.value[1]);
    REQUIRE(pypi.ok);
    CHECK(pypi.value.json.find("\"command\": \"uvx\"") != std::string::npos);
    CHECK(pypi.value.json.find("mcp-server-fetch@1.0.0") != std::string::npos);

    auto hosted = mcp_entry_for(all.value[2]);
    REQUIRE_FALSE(hosted.ok);
    CHECK(hosted.error.message.find("https://mcp.example.com") != std::string::npos);
    CHECK_FALSE(mcp_entry_for(all.value[3]).ok); // container images are not started
}
