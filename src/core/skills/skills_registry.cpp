#include <agent-cpp/agent.hpp>
#include <filesystem>
#include <fstream>
#include <simdjson.h>
#include <sstream>

namespace agent {

namespace {

// Parses the minimal SKILL.md frontmatter block:
//   ---
//   name: skill-name
//   description: one-line description
//   ---
// Only `name`/`description` are extracted; a full YAML parser isn't needed for two keys.
struct Frontmatter {
    std::string name;
    std::string description;
};

std::string_view trim(std::string_view s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) return {};
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

bool parse_frontmatter(std::string_view content, Frontmatter &out) {
    if (!content.starts_with("---")) return false;
    size_t body_start = content.find('\n', 3);
    if (body_start == std::string_view::npos) return false;
    size_t end = content.find("\n---", body_start);
    if (end == std::string_view::npos) return false;

    std::string_view block = content.substr(body_start + 1, end - body_start - 1);
    size_t pos = 0;
    while (pos < block.size()) {
        size_t line_end = block.find('\n', pos);
        if (line_end == std::string_view::npos) line_end = block.size();
        std::string_view line = block.substr(pos, line_end - pos);
        size_t colon = line.find(':');
        if (colon != std::string_view::npos) {
            std::string_view key = trim(line.substr(0, colon));
            std::string_view value = trim(line.substr(colon + 1));
            if (key == "name") out.name = std::string(value);
            else if (key == "description") out.description = std::string(value);
        }
        pos = line_end + 1;
    }
    return !out.name.empty() && !out.description.empty();
}

Result<std::string> skill_loader_callback(std::string_view arguments, void *user_data) {
    auto *session_cell = static_cast<Session **>(user_data);
    Session *session = (session_cell && *session_cell) ? *session_cell : nullptr;
    if (!session) return fail<std::string>(ErrorCode::InvalidConfig, "use_skill: session reference is null");

    std::string_view name_sv;
    simdjson::ondemand::parser parser;
    simdjson::padded_string json(arguments.data(), arguments.size());
    simdjson::ondemand::document doc;
    if (!parser.iterate(json).get(doc)) {
        simdjson::ondemand::object obj;
        if (!doc.get_object().get(obj)) {
            for (auto field : obj) {
                auto k = field.unescaped_key();
                if (!k.error() && k.value() == "name") (void)field.value().get_string().get(name_sv);
            }
        }
    }
    if (name_sv.empty()) return fail<std::string>(ErrorCode::InvalidConfig, "use_skill: missing 'name'");

    return read_skill_body(*session, name_sv);
}

} // namespace

Result<void> load_skills_dir(Session &session, std::string_view skills_dir) {
    std::error_code ec;
    if (!std::filesystem::is_directory(skills_dir, ec))
        return fail(ErrorCode::InvalidConfig, "skills_dir is not a directory: " + std::string(skills_dir));

    // Range-based for would use directory_iterator::operator++() (the throwing overload) on
    // every iteration, even though construction here is non-throwing -- advance manually with
    // the error_code overload so a mid-scan filesystem error can't throw an uncaught
    // std::filesystem::filesystem_error.
    for (auto it = std::filesystem::directory_iterator(skills_dir, ec); !ec && it != std::filesystem::directory_iterator();
         it.increment(ec)) {
        const auto &entry = *it;
        if (!entry.is_directory()) continue;

        std::filesystem::path skill_md = entry.path() / "SKILL.md";
        if (!std::filesystem::exists(skill_md)) continue;

        std::ifstream f(skill_md, std::ios::binary);
        if (!f) continue;
        std::stringstream buf;
        buf << f.rdbuf();
        std::string content = buf.str();

        Frontmatter fm;
        if (!parse_frontmatter(content, fm)) continue;

        session.skills[fm.name] = Skill{.name = fm.name, .description = fm.description, .path = entry.path().string()};
    }

    if (session.session_ptr_cell) {
        bool already_registered = false;
        for (const auto &t : session.config.tools)
            if (t.name == "use_skill") already_registered = true;

        if (!already_registered) {
            session.config.tools.push_back(Tool{
                .name = "use_skill",
                .description = "Load the full instructions for a named skill. Call this before "
                               "using a skill whose name matches the task at hand.",
                .parameter_schema = R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})",
                .callback = skill_loader_callback,
                .user_data = session.session_ptr_cell.get(),
            });
        }
    }

    rebuild_tool_index(session);
    return ok();
}

Result<std::string> read_skill_body(Session &session, std::string_view name) {
    auto it = session.skills.find(std::string(name));
    if (it == session.skills.end()) return fail<std::string>(ErrorCode::KeyNotFound, "Unknown skill: " + std::string(name));

    std::filesystem::path skill_md = std::filesystem::path(it->second.path) / "SKILL.md";
    std::ifstream f(skill_md, std::ios::binary);
    if (!f) return fail<std::string>(ErrorCode::FilesystemError, "Cannot open: " + skill_md.string());

    std::stringstream buf;
    buf << f.rdbuf();
    return ok(buf.str());
}

} // namespace agent
